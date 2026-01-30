"""pytorchexample: A Flower / PyTorch app."""

import random

import numpy as np
import torch
from flwr.app import Array, ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


def select_models(losses: list[float], thresh: float = 0.7):
    minloss = min(losses)
    maxloss = max(losses)
    deltaloss = maxloss - minloss

    # if deltaloss <= 1e-6:
    weights = [1] * len(losses)
    identities = [i for i in range(len(losses))]
    identities = random.choices(identities, k=1)
    return weights, identities

    # weights = [1 - (loss - minloss) / deltaloss for loss in losses]
    # identities = [i for i, w in enumerate(weights) if w >= thresh]
    # return weights, identities


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read context and run configs
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    num_global_models = int(context.run_config["num-global-models"])
    local_epochs = int(context.run_config["local-epochs"])

    # Load the data
    trainloader, evalloader = load_data(partition_id, num_partitions, batch_size)

    # Load each model and get the losses
    arrays_list = []
    losses = []
    for i in range(num_global_models):
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

    weights, identities = select_models(losses)

    fusion_np_arrays = {}
    for i in identities:
        arrays = arrays_list[i]
        weight = weights[i]
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
    batch_size = int(context.run_config["batch-size"])

    # Load the model and initialize it with the received weights
    # model = Net()
    # model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    # model.to(device)

    # Load the data
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    # eval_loss, eval_acc = test_fn(
    #     model,
    #     valloader,
    #     device,
    # )

    # Construct and return reply Message
    metric_record = MetricRecord(
        {
            "eval_loss": 0,
            "eval_acc": 0,
            "num-examples": len(valloader.dataset),
        }
    )
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
