"""
Treino, avaliação e persistência do modelo de regressão linear (Fases 5 e 6).
"""
import json
from pyexpat import model
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from src.config import MODEL_DIR, MODEL_FILE, METRICS_FILE


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """Treina uma Regressão Linear simples nos dados de treino."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: LinearRegression, X, y_true) -> dict:
    """Calcula MAE, MSE, RMSE e R2 das previsões do modelo."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def train_model_GradientBoosting(X_train: pd.DataFrame, y_train: pd.Series) -> GradientBoostingRegressor:
    """Treina um Gradiente Boosting Regressor nos dados de treino."""
    model = GradientBoostingRegressor(
    random_state=42,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3
)
    model.fit(X_train, y_train)
    return model

def evaluate_model_GradientBoosting(model: GradientBoostingRegressor, X, y_true) -> dict:
    """Calcula MAE, MSE, RMSE e R2 das previsões do modelo."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true, y_pred),
    }


def cross_validate_model(model, X, y) -> np.ndarray:
    """Realiza validação cruzada do modelo e retorna os scores de R2."""
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    return scores

def save_model(model: object) -> None:
    """Salva o modelo treinado em models/v1/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    print(f"Modelo salvo em: {MODEL_FILE}")


def save_metrics(metrics: dict) -> None:
    """Salva o dicionário de metadados/métricas em models/v1/."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Métricas salvas em: {METRICS_FILE}")
