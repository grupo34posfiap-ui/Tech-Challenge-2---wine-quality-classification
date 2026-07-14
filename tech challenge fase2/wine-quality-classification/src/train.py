"""
train.py
========
Treinamento, avaliação e comparação de modelos de classificação para
prever a qualidade do vinho (binária: alta vs baixa/média).

Modelos comparados:
  1. Regressão Logística (baseline linear, interpretável)
  2. Random Forest (ensemble de árvores, captura não-linearidades)
  3. Gradient Boosting (ensemble sequencial, costuma maximizar AUC)

Estratégia para o desbalanceamento (~14% positivos):
  - Split estratificado (mantém a proporção de classes em treino/teste).
  - class_weight="balanced" nos modelos que suportam.
  - Métricas focadas em ROC-AUC, F1 e recall da classe positiva
    (acurácia sozinha é enganosa com classes desbalanceadas).
  - Validação cruzada estratificada (5 folds) para estimativa robusta.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, classification_report,
)

from preprocessing import prepare

sns.set_theme(style="whitegrid")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RANDOM_STATE = 42
SPOT_RED = "#E3001B"


def build_models():
    """Retorna dict de pipelines. Scaler dentro do pipeline evita leakage."""
    return {
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
        ]),
        "Gradient Boosting": Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=3,
                random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


def main():
    X, y, df, feats = prepare("../data/WineQT.csv")
    print(f"Dataset final: {X.shape[0]} amostras, {X.shape[1]} features")
    print(f"Positivos (alta qualidade): {y.sum()} ({y.mean()*100:.1f}%)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    models = build_models()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    results = {}
    fitted = {}
    roc_data = {}
    for name, model in models.items():
        # validação cruzada (ROC-AUC) no treino
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate(model, X_test, y_test)
        metrics["cv_roc_auc_mean"] = cv_auc.mean()
        metrics["cv_roc_auc_std"] = cv_auc.std()
        results[name] = metrics
        fitted[name] = model
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = (fpr, tpr, metrics["roc_auc"])
        print(f"--- {name} ---")
        print(f"  CV ROC-AUC: {cv_auc.mean():.3f} +/- {cv_auc.std():.3f}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            print(f"  {k:10s}: {metrics[k]:.3f}")
        print()

    # ---- Tabela comparativa ----
    comp = pd.DataFrame(results).T[
        ["accuracy", "precision", "recall", "f1", "roc_auc",
         "cv_roc_auc_mean", "cv_roc_auc_std"]]
    comp = comp.sort_values("roc_auc", ascending=False)
    comp.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"))
    print("Comparação salva em results/model_comparison.csv")
    print(comp.round(3).to_string())

    best_name = comp.index[0]
    best_model = fitted[best_name]
    print(f"\n>>> Melhor modelo: {best_name} (ROC-AUC teste = {comp.loc[best_name,'roc_auc']:.3f})")

    # ---- Gráfico comparativo de métricas ----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    comp_plot = comp[metric_cols]
    comp_plot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="black", width=.8)
    ax.set_title("Comparação de desempenho entre modelos", fontweight="bold")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.05)
    ax.set_xticklabels(comp_plot.index, rotation=0)
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=6, padding=1)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "07_model_comparison.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Curvas ROC ----
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, auc) in roc_data.items():
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório")
    ax.set_xlabel("Taxa de Falsos Positivos"); ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.set_title("Curvas ROC", fontweight="bold"); ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "08_roc_curves.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Matriz de confusão do melhor modelo ----
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
                xticklabels=["Baixa/Média", "Alta"],
                yticklabels=["Baixa/Média", "Alta"], ax=ax)
    ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusão — {best_name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "09_confusion_matrix.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(RESULTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Melhor modelo: {best_name}\n\n")
        f.write(classification_report(y_test, y_pred_best,
                target_names=["Baixa/Média", "Alta"]))

    # ---- Importância de variáveis (permutation, agnóstica ao modelo) ----
    perm = permutation_importance(best_model, X_test, y_test, n_repeats=30,
                                  random_state=RANDOM_STATE, scoring="roc_auc")
    imp = pd.Series(perm.importances_mean, index=feats).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [SPOT_RED if v > 0 else "#9aa0a6" for v in imp.values]
    ax.barh(imp.index, imp.values, color=colors)
    ax.set_title("Importância das variáveis (Permutation Importance, métrica=ROC-AUC)",
                 fontweight="bold", fontsize=11)
    ax.set_xlabel("Queda média no ROC-AUC ao embaralhar a variável")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "10_feature_importance.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    imp.sort_values(ascending=False).to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"))

    # ---- Persistir métricas em JSON ----
    summary = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "pct_high_quality": float(y.mean()),
        "best_model": best_name,
        "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in results.items()},
        "top_features": imp.sort_values(ascending=False).head(8).round(4).to_dict(),
    }
    with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nTop 5 variáveis mais importantes:")
    print(imp.sort_values(ascending=False).head(5).round(4).to_string())
    print("\nTodos os artefatos salvos em results/")
    return summary


if __name__ == "__main__":
    main()
