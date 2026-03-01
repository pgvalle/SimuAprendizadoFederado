import matplotlib.pyplot as plt
import pandas as pd
import os

# Configuração de estilo acadêmico
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.figsize": (10, 6)
})

def load_and_process(filepath):
    df = pd.read_csv(filepath)
    # Agrupa por rodada e calcula média e desvio padrão para todas as métricas
    # O Pandas criará um MultiIndex nas colunas: (métrica, estatística)
    stats = df.groupby("round").agg(["mean", "std"])
    return stats

def plot_metric(flasc_stats, flsc_stats, metric, ylabel, title, filename):
    plt.figure()
    
    rounds = flasc_stats.index
    
    # Plot FLASC
    plt.errorbar(
        rounds,
        flasc_stats[metric]["mean"],
        yerr=flasc_stats[metric]["std"],
        fmt="-o",
        capsize=5,
        label="FLASC",
        color="blue",
        alpha=0.8
    )
    
    # Plot FLSC
    plt.errorbar(
        rounds,
        flsc_stats[metric]["mean"],
        yerr=flsc_stats[metric]["std"],
        fmt="-s",
        capsize=5,
        label="FLSC",
        color="red",
        alpha=0.8
    )
    
    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rounds)
    plt.legend()
    
    output_path = os.path.join("article/figs", filename)
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"Gráfico salvo: {output_path}")

if __name__ == "__main__":
    # Garante que o diretório de figuras existe
    os.makedirs("article/figs", exist_ok=True)
    
    # Carrega dados
    flasc = load_and_process("results/flasc-1.csv")
    flsc = load_and_process("results/flsc-1.csv")
    
    # Gera os três gráficos
    plot_metric(flasc, flsc, "train_loss", "Loss", "Training Loss comparison", "train_loss.svg")
    plot_metric(flasc, flsc, "eval_loss", "Loss", "Evaluation Loss comparison", "eval_loss.svg")
    plot_metric(flasc, flsc, "eval_acc", "Accuracy", "Evaluation Accuracy comparison", "eval_accuracy.svg")
