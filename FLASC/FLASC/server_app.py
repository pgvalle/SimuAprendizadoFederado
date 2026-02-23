"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from common.strategy import ClusterStrategy
from common.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train = float(context.run_config["fraction-train"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    num_rounds = int(context.run_config["num-server-rounds"])
    num_models = int(context.run_config["num-global-models"])
    rho = float(context.run_config["rho"])
    print(f"rho={rho}")

    # Initialize FedAvg strategy
    strategy = ClusterStrategy(
        num_models=num_models,
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
    )

    # Start strategy, run FedAvg for `num_rounds`
    strategy.my_start(grid=grid, num_rounds=num_rounds)


# def global_evaluate(server_round: int, arrays: ArrayRecord, seed: int) -> MetricRecord:
#     """Evaluate model on central data."""

#     # Load the model and initialize it with the received weights
#     model = Net()
#     model.load_state_dict(arrays.to_torch_state_dict())
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model.to(device)

#     # Load entire test set
#     test_dataloader = load_centralized_dataset(seed)

#     # Evaluate the global model on the test set
#     test_loss, test_acc = test(model, test_dataloader, device)

#     # Return the evaluation metrics
#     return MetricRecord({"accuracy": test_acc, "loss": test_loss})
