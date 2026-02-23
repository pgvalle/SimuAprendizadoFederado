"""pytorchexample: A Flower / PyTorch app."""

import random

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from common.task import Net, load_data
from common.task import test as test_fn
from common.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def select_models(losses: list[float], rho: float):
    count = len(losses)
    lmax, lmin = max(losses), min(losses)
    lrange = lmax - lmin

    # Case: All losses are identical or extremely close
    if lrange <= 1e-5:
        return [np.random.choice(range(count))], [1.0]

    # 1. Normalize and Filter in one pass
    # we use (1 - normalized_loss) because lower loss should have higher weight
    selected_indices = []
    raw_weights = []

    for i, loss in enumerate(losses):
        norm_l = (loss - lmin) / lrange
        if norm_l <= rho:
            selected_indices.append(i)
            raw_weights.append(1.0 - norm_l)

    if not selected_indices:
        # Fallback if rho is so small nothing was selected
        idx = losses.index(lmin)
        return [idx], [1.0]

    # 2. Apply "Value as Weight" logic
    total_raw_weight = sum(raw_weights)
    final_weights = [w / total_raw_weight for w in raw_weights]
    return selected_indices, final_weights


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read context and run configs
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    local_epochs = int(context.run_config["local-epochs"])
    num_models = int(context.run_config["num-global-models"])
    rho = float(context.run_config["rho"])

    # Load the data
    trainloader, evalloader = load_data(partition_id, num_partitions, batch_size)

    # Load each model and get the losses
    arrays_list = []
    losses = []
    for i in range(num_models):
        # get ith cluster model
        arrays = msg.content[f"{i}"]
        arrays_list.append(arrays)

        # load into pytorch to evaluate
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        model.eval()

        # Get loss for local data and store it
        eval_loss, _ = test_fn(model, evalloader, device)
        losses.append(eval_loss)

    identities, weights = select_models(losses, rho)
    fusion_np_arrays = {}
    for i in identities:
        arrays = arrays_list[i]
        weight = weights.pop(0)

        for k, v in arrays.items():
            if k not in fusion_np_arrays:
                fusion_np_arrays[k] = v.numpy() * weight
            else:
                fusion_np_arrays[k] += v.numpy() * weight

    # load fusion as pytorch model
    fusion_arrays = ArrayRecord(
        {k: Array(np.asarray(v)) for k, v in fusion_np_arrays.items()}
    )
    fusion = Net()
    fusion.load_state_dict(fusion_arrays.to_torch_state_dict())

    train_loss = train_fn(
        fusion,
        trainloader,
        local_epochs,
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    fusion_record = ArrayRecord(fusion.state_dict())
    metric_record = MetricRecord(
        {
            "train_loss": train_loss,
            "identities": identities,
            "num-examples": len(trainloader.dataset),
        }
    )
    content = RecordDict({"arrays": fusion_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read context and run configs
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    num_models = int(context.run_config["num-global-models"])
    batch_size = int(context.run_config["batch-size"])

    # Load the data
    _, evalloader = load_data(partition_id, num_partitions, batch_size)

    best_eval_loss = 1e10
    best_eval_acc = 0
    for i in range(num_models):
        arrays = msg.content[f"{i}"]
        model = Net()
        model.load_state_dict(arrays.to_torch_state_dict())
        model.to(device)
        model.eval()

        eval_loss, eval_acc = test_fn(model, evalloader, device)
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_eval_acc = eval_acc

    # Construct and return reply Message
    metric_record = MetricRecord(
        {
            "eval_loss": best_eval_loss,
            "eval_acc": best_eval_acc,
            "num-examples": len(evalloader.dataset),
        }
    )
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
