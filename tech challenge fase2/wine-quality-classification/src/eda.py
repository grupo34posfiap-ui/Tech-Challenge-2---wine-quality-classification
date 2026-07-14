"""
eda.py
======
Análise Exploratória de Dados (EDA) do Wine Quality Dataset.
Gera todos os gráficos e estatísticas usados no storytelling executivo.
Os artefatos são salvos em ../results/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import load_data, binarize_target, remove_duplicates, FEATURE_COLS

sns.set_theme(style="whitegrid")
PALETTE = {"baixa": "#9aa0a6", "alta": "#E3001B"}  # cinza vs vermelho SPOT
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] salvo: {name}")


def run_eda(path: str = "../data/WineQT.csv"):
    df_raw = load_data(path)
    df = binarize_target(df_raw)
    df = remove_duplicates(df)

    # ---- 1. Balanceamento das classes ----
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df["high_quality"].value_counts().sort_index()
    bars = ax.bar(["Baixa/Média\n(nota < 7)", "Alta\n(nota >= 7)"], counts.values,
                  color=[PALETTE["baixa"], PALETTE["alta"]])
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width()/2, v + 5, f"{v}\n({v/len(df)*100:.1f}%)",
                ha="center", va="bottom", fontweight="bold")
    ax.set_title("Desbalanceamento de classes\nApenas ~14% dos vinhos são de alta qualidade",
                 fontweight="bold")
    ax.set_ylabel("Nº de amostras")
    _save(fig, "01_class_balance.png")

    # ---- 2. Distribuição original da nota ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    q = df["quality"].value_counts().sort_index()
    colors = [PALETTE["alta"] if k >= 7 else PALETTE["baixa"] for k in q.index]
    ax.bar(q.index.astype(str), q.values, color=colors)
    ax.set_title("Distribuição das notas de qualidade (3–8)\nConcentração esmagadora em 5 e 6",
                 fontweight="bold")
    ax.set_xlabel("Nota atribuída por especialistas")
    ax.set_ylabel("Nº de amostras")
    _save(fig, "02_quality_distribution.png")

    # ---- 3. Matriz de correlação ----
    fig, ax = plt.subplots(figsize=(11, 9))
    corr = df[FEATURE_COLS + ["quality"]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, cbar_kws={"shrink": .8}, ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Matriz de correlação (Pearson)", fontweight="bold", pad=12)
    _save(fig, "03_correlation_matrix.png")

    # ---- 4. Correlação de cada feature com a qualidade ----
    fig, ax = plt.subplots(figsize=(7, 5))
    corr_q = corr["quality"].drop("quality").sort_values()
    colors = [PALETTE["alta"] if v > 0 else PALETTE["baixa"] for v in corr_q.values]
    ax.barh(corr_q.index, corr_q.values, color=colors)
    ax.axvline(0, color="black", lw=.8)
    ax.set_title("Correlação de cada variável com a qualidade",
                 fontweight="bold")
    ax.set_xlabel("Coeficiente de Pearson")
    _save(fig, "04_corr_with_quality.png")

    # ---- 5. Boxplots das variáveis mais relevantes por classe ----
    key_vars = ["alcohol", "volatile acidity", "sulphates", "citric acid",
                "density", "total sulfur dioxide"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, var in zip(axes.ravel(), key_vars):
        sns.boxplot(data=df, x="high_quality", y=var, ax=ax,
                    palette=[PALETTE["baixa"], PALETTE["alta"]], width=.6)
        ax.set_xticklabels(["Baixa/Média", "Alta"])
        ax.set_xlabel("")
        ax.set_title(var, fontweight="bold")
    fig.suptitle("Distribuição das variáveis-chave por classe de qualidade",
                 fontweight="bold", fontsize=14)
    fig.tight_layout()
    _save(fig, "05_boxplots_by_class.png")

    # ---- 6. Outliers: contagem via IQR ----
    outlier_counts = {}
    for col in FEATURE_COLS:
        q1, q3 = df[col].quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
        outlier_counts[col] = int(((df[col] < lo) | (df[col] > hi)).sum())
    oc = pd.Series(outlier_counts).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(oc.index, oc.values, color="#E3001B")
    for i, v in enumerate(oc.values):
        ax.text(v + 1, i, f"{v} ({v/len(df)*100:.1f}%)", va="center", fontsize=8)
    ax.set_title("Outliers por variável (regra do IQR 1.5×)", fontweight="bold")
    ax.set_xlabel("Nº de amostras atípicas")
    _save(fig, "06_outliers.png")

    # ---- Resumo estatístico em CSV ----
    desc = df[FEATURE_COLS + ["quality"]].describe().T
    desc["outliers_iqr"] = pd.Series(outlier_counts)
    desc.to_csv(os.path.join(RESULTS_DIR, "eda_summary_stats.csv"))
    print("[eda] eda_summary_stats.csv salvo")

    # Insights numéricos para o storytelling
    print("\n=== INSIGHTS-CHAVE ===")
    print("Top correlações |r| com quality:")
    print(corr["quality"].drop("quality").abs().sort_values(ascending=False).head(5))
    print("\nMédia de álcool: alta=%.2f vs baixa=%.2f" % (
        df[df.high_quality==1]["alcohol"].mean(),
        df[df.high_quality==0]["alcohol"].mean()))
    print("Média acidez volátil: alta=%.3f vs baixa=%.3f" % (
        df[df.high_quality==1]["volatile acidity"].mean(),
        df[df.high_quality==0]["volatile acidity"].mean()))
    return df


if __name__ == "__main__":
    run_eda()
