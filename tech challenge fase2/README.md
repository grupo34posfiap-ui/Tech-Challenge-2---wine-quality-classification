# Wine Quality Classification 

**Tech Challenge — Fase 2 | POSTECH Data Analytics**

Pipeline completa de Machine Learning para classificar a qualidade de vinhos a
partir de suas características físico-químicas, transformando o problema em uma
**classificação binária**:

- 🔴 **Alta Qualidade** — nota ≥ 7
- ⚪ **Baixa/Média Qualidade** — nota < 7

---

##  Resultado principal

| Modelo | ROC-AUC | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | **0,916** | 0,90 | 0,71 | 0,44 | 0,55 |
| Gradient Boosting | 0,886 | 0,88 | 0,57 | 0,44 | 0,50 |
| Regressão Logística | 0,884 | 0,79 | 0,36 | 0,74 | 0,48 |

> O **Random Forest** foi o melhor modelo geral (maior ROC-AUC e precisão). A
> **Regressão Logística** se destaca no recall — útil quando o objetivo é não
> deixar passar nenhum vinho premium.

### Variáveis mais influentes na qualidade
1. **Sulfatos** (conservação/antioxidação)
2. **Acidez volátil** (quanto menor, melhor — evita o defeito "avinagrado")
3. **Álcool** (maior teor → melhor avaliação)

---

## Estrutura do repositório

```
wine-quality-classification/
├── data/
│   └── WineQT.csv                      # Base de dados (Wine Quality Dataset)
├── notebooks/
│   └── wine_quality_analysis.ipynb     # Notebook completo (EDA + modelagem)
├── src/
│   ├── preprocessing.py                # Carga, binarização, dedup, feature engineering
│   ├── eda.py                          # Análise exploratória (gera os gráficos)
│   └── train.py                        # Treino, avaliação e comparação dos modelos
├── results/
│   ├── *.png                           # Gráficos da EDA e da modelagem
│   ├── model_comparison.csv            # Tabela comparativa de métricas
│   ├── feature_importance.csv          # Importância das variáveis
│   ├── classification_report.txt       # Relatório do melhor modelo
│   └── metrics_summary.json            # Resumo consolidado em JSON
├── apresentacao_executiva.pdf          # Storytelling executivo da análise
├── requirements.txt
└── README.md
```

---

## Como executar

```bash
# 1. Criar ambiente e instalar dependências
pip install -r requirements.txt

# 2. Rodar a EDA (gera os gráficos em results/)
cd src
python eda.py

# 3. Treinar e avaliar os modelos
python train.py

# Ou abrir o notebook completo:
jupyter notebook notebooks/wine_quality_analysis.ipynb
```

---

## Metodologia

1. **Compreensão do problema** — binarização da nota de qualidade.
2. **EDA** — distribuições, correlações justificadas, detecção de outliers (IQR),
   análise de balanceamento (~14% de positivos → classe desbalanceada).
3. **Pré-processamento** — sem nulos; remoção de 125 duplicatas (evita *leakage*);
   padronização via `StandardScaler` dentro do pipeline; 4 novas features com
   justificativa enológica.
4. **Modelagem** — 3 algoritmos com split estratificado e `class_weight='balanced'`.
5. **Avaliação** — ROC-AUC, F1, precisão, recall, matriz de confusão e validação
   cruzada estratificada (5 folds).
6. **Interpretação** — *permutation importance* + recomendações para a produção.

---

## Decisões técnicas relevantes

- **Duplicatas removidas** considerando apenas as features (não o `Id`), para evitar
  que a mesma amostra apareça em treino e teste (vazamento de dados).
- **Outliers mantidos** — representam variação física real; modelos de árvore são robustos.
- **Métricas além da acurácia** — com classes desbalanceadas, acurácia é enganosa;
  o foco é ROC-AUC e F1.
- **Scaler dentro do pipeline** — a padronização é ajustada apenas no treino.

---

## Dataset

Wine Quality Dataset (Kaggle) — 1.143 amostras, 11 variáveis físico-químicas +
nota de qualidade. Após limpeza: 1.018 amostras únicas.
