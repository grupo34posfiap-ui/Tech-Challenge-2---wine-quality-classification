"""
preprocessing.py
================
Funções de carga e preparação dos dados do Wine Quality Dataset.

Decisões de projeto:
- A coluna `Id` é descartada (identificador, sem valor preditivo).
- Linhas duplicadas (idênticas em todas as features) são removidas para
  evitar vazamento (data leakage) entre treino e teste.
- A variável alvo `quality` é binarizada: 1 = Alta Qualidade (nota >= 7),
  0 = Baixa/Média Qualidade (nota < 7), conforme o enunciado.
- Feature engineering: criação de razões físico-químicas interpretáveis.
"""

import pandas as pd
import numpy as np

QUALITY_THRESHOLD = 7

FEATURE_COLS = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]


def load_data(path: str) -> pd.DataFrame:
    """Carrega o CSV e remove a coluna identificadora `Id` se existir."""
    df = pd.read_csv(path)
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
    return df


def binarize_target(df: pd.DataFrame, threshold: int = QUALITY_THRESHOLD) -> pd.DataFrame:
    """Cria a coluna binária `high_quality` (1 se quality >= threshold)."""
    df = df.copy()
    df["high_quality"] = (df["quality"] >= threshold).astype(int)
    return df


def remove_duplicates(df: pd.DataFrame, subset=None) -> pd.DataFrame:
    """Remove linhas duplicadas considerando apenas as features (evita leakage)."""
    if subset is None:
        subset = FEATURE_COLS
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = before - len(df)
    print(f"[preprocessing] Duplicatas removidas: {removed} ({before} -> {len(df)})")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria novas features interpretáveis a partir das variáveis originais.
    Todas têm justificativa enológica:
    - free_sulfur_ratio: fração do SO2 que está livre (poder antioxidante ativo).
    - total_acidity: soma das acidezes (estrutura ácida total).
    - alcohol_density_ratio: proxy de corpo/concentração do vinho.
    - sulphates_chlorides_ratio: equilíbrio entre conservante e sais.
    """
    df = df.copy()
    eps = 1e-6
    df["free_sulfur_ratio"] = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + eps)
    df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"] + df["citric acid"]
    df["alcohol_density_ratio"] = df["alcohol"] / (df["density"] + eps)
    df["sulphates_chlorides_ratio"] = df["sulphates"] / (df["chlorides"] + eps)
    return df


def prepare(path: str, use_feature_engineering: bool = True,
            drop_duplicates: bool = True):
    """
    Pipeline completa de preparação. Retorna (X, y, df_completo, feature_names).
    """
    df = load_data(path)
    df = binarize_target(df)
    if drop_duplicates:
        df = remove_duplicates(df)
    if use_feature_engineering:
        df = engineer_features(df)

    feature_names = [c for c in df.columns if c not in ("quality", "high_quality")]
    X = df[feature_names].copy()
    y = df["high_quality"].copy()
    return X, y, df, feature_names


if __name__ == "__main__":
    X, y, df, feats = prepare("../data/WineQT.csv")
    print("Shape X:", X.shape)
    print("Features:", feats)
    print("Distribuição alvo:\n", y.value_counts())
