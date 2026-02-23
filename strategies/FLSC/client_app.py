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


def select_models(losses: list[float], n: int):
    count = len(losses)
    lmax = max(losses)
    lmin = min(losses)
    lrange = lmax - lmin

    if lrange <= 1e-5:
        return np.random.choice(range(count), size=n, replace=False).tolist()

    ilosses = [(i, losses[i]) for i in range(count)]
    ilosses.sort(key=lambda iloss: iloss[1])
    return [ilosses[i][0] for i in range(n)]


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Read context and run configs
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    batch_size = int(context.run_config["batch-size"])
    num_models = int(context.run_config["num-global-models"])
    n = int(context.run_config["n"])

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

    identities = select_models(losses, n)
    fusion_np_arrays = {}
    for i in identities:
        arrays = arrays_list[i]
        for k, v in arrays.items():
            if k not in fusion_np_arrays:
                fusion_np_arrays[k] = v.numpy() / n
            else:
                fusion_np_arrays[k] += v.numpy() / n

    # load fusion as pytorch model
    fusion_arrays = ArrayRecord(
        {k: Array(np.asarray(v)) for k, v in fusion_np_arrays.items()}
    )
    fusion = Net()
    fusion.load_state_dict(fusion_arrays.to_torch_state_dict())

    train_loss = train_fn(
        fusion,
        trainloader,
        int(context.run_config["local-epochs"]),
        float(context.run_config["learning-rate"]),
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
