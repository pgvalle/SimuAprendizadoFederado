from collections.abc import Callable, Iterable
from logging import INFO
from typing import cast

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid
from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    sample_nodes,
)

from pytorchexample.task import Net


def my_aggregate_metricrecords(records: list[RecordDict]) -> MetricRecord:
    """Perform weighted aggregation all MetricRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        # Because replies have been checked for consistency,
        # we can safely cast the weighting factor to float
        w = cast(float, metricrecord["num-examples"])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, weight_factors, strict=True):
        for record_item in record.metric_records.values():
            # aggregate in-place
            for key, value in record_item.items():
                if key in ["num-examples", "identities", "identity"]:
                    # We exclude the weighting key from the aggregated MetricRecord
                    continue
                if key not in aggregated_metrics:
                    if isinstance(value, list):
                        aggregated_metrics[key] = [v * weight for v in value]
                    else:
                        aggregated_metrics[key] = value * weight
                else:
                    if isinstance(value, list):
                        current_list = cast(list[float], aggregated_metrics[key])
                        aggregated_metrics[key] = [
                            curr + val * weight
                            for curr, val in zip(current_list, value, strict=True)
                        ]
                    else:
                        current_value = cast(float, aggregated_metrics[key])
                        aggregated_metrics[key] = current_value + value * weight

    return aggregated_metrics


class FLSC(FedAvg):
    def __init__(
        self,
        num_models: int,
        fraction_train: float = 0.5,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
    ) -> None:
        super().__init__(
            fraction_train=fraction_train,
            fraction_evaluate=fraction_evaluate,
            min_train_nodes=min_train_nodes,
            min_evaluate_nodes=min_evaluate_nodes,
            min_available_nodes=min_available_nodes,
        )

        self.arrays_dict = {}
        for i in range(num_models):
            model = Net()
            state = model.state_dict()
            arrays = ArrayRecord(torch_state_dict=state)
            self.arrays_dict.update({f"{i}": arrays})

    def update_clusters(self, records: list[RecordDict]) -> None:
        num_global_models = len(self.arrays_dict)

        # Group RecordDicts to form the clusters
        clusters = [[] for _ in range(num_global_models)]
        for record in records:
            metrics = next(iter(record.metric_records.values()))
            identities = metrics["identities"]
            # add this client RecordDict to all clusters it belongs to
            for i in identities:
                clusters[i].append(record)

        # Do FedAvg on each cluster that has members
        for i in range(num_global_models):
            if clusters[i] == []:
                log(INFO, f"cluster {i} has no update!")
            else:
                self.arrays_dict[f"{i}"] = aggregate_arrayrecords(
                    clusters[i], "num-examples"
                )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""

        # Do not configure federated train if fraction_train is 0.
        if self.fraction_train == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict()
        record.update(self.arrays_dict)
        record.update({self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        metrics = None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]
            # Update clusters
            self.update_clusters(reply_contents)
            # Aggregate MetricRecords
            metrics = my_aggregate_metricrecords(reply_contents)
        return None, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Do not configure federated evaluation if fraction_evaluate is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        record = RecordDict()
        record.update(self.arrays_dict)
        record.update({self.configrecord_key: config})
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def my_start(
        self,
        grid: Grid,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
    ) -> Result:
        return super().start(
            grid,
            ArrayRecord(),
            num_rounds,
            timeout,
            train_config,
            evaluate_config,
            evaluate_fn,
        )
