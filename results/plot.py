import matplotlib.pyplot as plt
import pandas as pd
import os

# Configuração de estilo acadêmico (Fontes aumentadas para melhor legibilidade em figuras pequenas)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 20,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "figure.figsize": (8, 5)
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
        linewidth=2,
        markersize=8,
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
        linewidth=2,
        markersize=8,
        label="FLSC",
        color="red",
        alpha=0.8
    )
    
    plt.title(title)
    plt.xlabel("Rodada")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(rounds)
    plt.legend()
    
    output_path = os.path.join("article/figs", filename)
    plt.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {output_path}")

if __name__ == "__main__":
    # Garante que o diretório de figuras existe
    results_dir = "results"
    output_dir = "article/figs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Identifica todos os sufixos disponíveis (ex: 1, 2, 3)
    suffixes = sorted(list(set(
        f.split('-')[-1].replace('.csv', '') 
        for f in os.listdir(results_dir) 
        if f.startswith('flasc-') and f.endswith('.csv')
    )))
    
    print(f"Encontrados experimentos: {suffixes}")
    
    # Parâmetros dos experimentos conforme definido no artigo
    params = {
        "1": r"$\rho=0, N=1$",
        "2": r"$\rho=0.5, N=2$",
        "3": r"$\rho=1, N=4$"
    }
    
    for suffix in suffixes:
        flasc_path = os.path.join(results_dir, f"flasc-{suffix}.csv")
        flsc_path = os.path.join(results_dir, f"flsc-{suffix}.csv")
        
        if os.path.exists(flasc_path) and os.path.exists(flsc_path):
            print(f"\nGerando gráficos para experimento {suffix}...")
            # Carrega dados
            flasc = load_and_process(flasc_path)
            flsc = load_and_process(flsc_path)
            
            # Formata o sufixo do título com os parâmetros
            p_text = params.get(suffix, f"Exp {suffix}")
            full_suffix = f"({p_text})"
            
            # Gera os três gráficos para este sufixo
            plot_metric(flasc, flsc, "train_loss", "Perda", f"Perda de Treinamento {full_suffix}", f"train_loss_{suffix}.pdf")
            plot_metric(flasc, flsc, "eval_loss", "Perda", f"Perda de Avaliação {full_suffix}", f"eval_loss_{suffix}.pdf")
            plot_metric(flasc, flsc, "eval_acc", "Acurácia", f"Acurácia de Avaliação {full_suffix}", f"eval_accuracy_{suffix}.pdf")
        else:
            print(f"Aviso: Par de arquivos para sufixo {suffix} não encontrado.")
