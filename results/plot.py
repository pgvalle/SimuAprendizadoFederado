import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set font for academic style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10
})

# Load data
filename = input("filename: ")
df = pd.read_csv(filename)


# Calculate statistics per round
def get_stats(group):
    mean = group.mean()
    std = group.std(ddof=1)
    return pd.Series({"mean": mean, "std": std})


# Group by Round and calculate for each metric
stats_acc = df.groupby("Round")["Eval Acc"].apply(get_stats).unstack()
stats_loss = df.groupby("Round")["Eval Loss"].apply(get_stats).unstack()

# Plot Evaluation Accuracy
plt.figure(figsize=(10, 6))
plt.errorbar(
    stats_acc.index,
    stats_acc["mean"],
    yerr=stats_acc["std"],
    fmt="-o",
    capsize=5,
    label="Mean Eval Acc (± Std Dev)",
)
plt.title("Evaluation Accuracy per Round (Mean ± Std Dev)")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(stats_acc.index)
plt.legend()
plt.savefig("article/figs/eval_accuracy.svg")
plt.close()

# Plot Evaluation Loss
plt.figure(figsize=(10, 6))
plt.errorbar(
    stats_loss.index,
    stats_loss["mean"],
    yerr=stats_loss["std"],
    fmt="-o",
    color="red",
    capsize=5,
    label="Mean Eval Loss (± Std Dev)",
)
plt.title("Evaluation Loss per Round (Mean ± Std Dev)")
plt.xlabel("Round")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.7)
plt.xticks(stats_loss.index)
plt.legend()
plt.savefig("article/figs/eval_loss.svg")
plt.close()

print("Graphs saved: article/figs/eval_accuracy.svg and article/figs/eval_loss.svg")
