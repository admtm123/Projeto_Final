# Modelo de Previsão de Preços de Imóveis com Gradient Boosting

Projeto didático de ciência de dados que demonstra um pipeline completo de machine learning para regressão: EDA, limpeza de dados, feature engineering, preparação para modelagem, treino, avaliação e versionamento.

**Base de dados:** King County House Sales (Seattle, EUA) — 21.613 vendas de imóveis (2014–2015)  
**Modelo:** Gradient Boosting Regressor (`GradientBoostingRegressor` do scikit-learn) sobre 23 variáveis (features físicas do imóvel + faixa de preço do zipcode, com escalonamento via `StandardScaler`), prevendo `price` diretamente em USD  
**R² (teste):** ~0.8658 · **MAE (teste):** ~US$ 77,467

---

## Resultados principais

| Métrica | Valor |
|--------|-------|
| Modelo | Gradient Boosting Regressor |
| R² (teste) | 0,8658 |
| MAE (teste) | US$ 77.467 |
| Desempenho | Melhor do que a regressão linear neste projeto |

---

## Pipeline

| Fase | Descrição | Resultado |
|------|-----------|-----------|
| **Extração** | Download do dataset público | `data/raw/kc_house_data.csv` |
| **1 — EDA** | Distribuição de `price`, dispersão das variáveis com o alvo, mapa de correlação | figuras em `outputs/figures/` |
| **2 — Limpeza** | Remove duplicatas completas, imputa `sqft_above` ausente pela mediana, trata outlier de `bedrooms` (33 quartos) com winsorização por IQR (`clean_data`, `cap_iqr` em `src/features.py`) |
| **3 — Feature engineering** | Cria `ano_venda`, `idade_imovel`, `foi_reformado`; calcula `preco_por_m2` só para leitura (não usamos como preditora, pois deriva de `price`) (`add_features` em `src/features.py`); dataset resultante é salvo em disco (`save_processed_data` em `src/dataset.py`) | `data/processed/kc_house_processed.csv` |
| **4 — Preparação para modelagem** | Seleciona as 18 variáveis explicativas/features + variável-alvo (`select_final_columns`) e salva o recorte em na pasta data/final (`save_final_data`); a modelagem recarrega esse arquivo do zero (`load_final_data`) antes de continuar. Split treino/teste (80/20) *antes* do encoding; `zipcode` agrupado em 10 faixas de preço médio + one-hot encoding(`encode_zipcode` em `src/features.py`); remoção de colunas redundantes (`sqft_above`, `sqft_living15`, `sqft_lot15`) via análise de VIF; `StandardScaler` | `data/final/kc_house_final.csv` → matriz de 23 features escalonadas |
| **5 — Modelagem** | `GradientBoostingRegressor` do scikit-learn treinado nas features escalonadas; diagnóstico de overfitting (treino x teste) (`src/modeling/train.py`) | modelo treinado |
| **6 — Avaliação e versionamento** | MAE/MSE/RMSE/R² calculados no split 80/20 (estimativa honesta de erro); gráficos de real x previsto e resíduos; **retreino final do modelo com 100% dos dados** (treino + teste, já que não há mais necessidade de reservar uma fatia para teste) antes de salvar; modelo retreinado e métricas do split salvos como v1 | `models/v1/`, `outputs/figures/` |

---

## Estrutura do projeto

```
├── data/
│   ├── raw/                        
│   ├── processed/                  
│   └── final/                      
│
├── models/
│   └── v1/
│       ├── modelo_gradientBoosting_v1.pkl 
│       └── metricas_v1.json        
│
├── notebooks/
│   └── dataview_precos.ipynb      
│
├── outputs/
│   └── figures/                    
│
├── src/
│   ├── config.py                   
│   ├── dataset.py                  
│   ├── features.py                 
│   ├── plots.py                    
│   └── modeling/
│       └── train.py                
│
└── requirements.txt
```

---

## Como executar

```bash
# 1. Instalar dependências
#O arqivos requirements.txt não está configurado com as versões das bibliotecas de forma fixa, com isso existe a flexibilidade do 'pip' instalar as versões mais atualizadas e adequadas conforme a versão do Python.
pip install -r requirements.txt

# 2. Abrir o notebook
jupyter notebook notebooks/dataview_precos.ipynb
```

---

## Melhorias futuras

- **Feature engineering:** criar uma feature geoespacial, como `distancia_centro_cidade`, combinando `lat` e `long`, para potencialmente melhorar a capacidade preditiva do modelo.
- **Empacotar o pré-processamento com o modelo:** hoje `models/v1/modelo_gradientBoosting_v1.pkl` salva apenas o modelo treinado. O `StandardScaler` e o mapeamento de `zipcode` para faixas (`fit_zipcode_faixas`/`apply_zipcode_faixas` em `src/features.py`) não são persistidos junto, o que exige reprocessar os dados novos manualmente. Uma evolução natural seria migrar para um `sklearn.Pipeline` ou `ColumnTransformer` para encapsular pré-processamento e modelo em um único objeto.
- **Ajuste de hiperparâmetros:** testar diferentes configurações do `GradientBoostingRegressor` para buscar melhor equilíbrio entre precisão e generalização.
