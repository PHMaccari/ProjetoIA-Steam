"""
Predição e explicabilidade das decisões do modelo.

Implementa o fluxo: Entrada → IA → Resultado → Explicação
- Classifica o potencial de sucesso (3 classes)
- Retorna probabilidades por classe
- Gera gráfico e texto explicativo para apoio à decisão
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, MODELS_DIR, SUCCESS_LABELS


# Tradução das features técnicas para rótulos amigáveis na interface
FEATURE_LABELS_PT = {
    "price_usd": "Preço (USD)",
    "is_free": "Jogo gratuito",
    "num_languages": "Quantidade de idiomas",
    "achievement_count": "Número de achievements",
    "release_year": "Ano de lançamento",
    "release_month": "Mês de lançamento",
    "mat_supports_windows": "Suporte Windows",
    "mat_supports_mac": "Suporte Mac",
    "mat_supports_linux": "Suporte Linux",
    "publisher_tier": "Tier do publisher",
    "has_controller": "Suporte a controle",
    "num_categories": "Quantidade de tags/categorias",
    "cat_multiplayer": "Multiplayer",
    "cat_singleplayer": "Single-player",
    "cat_achievements": "Achievements Steam",
    "cat_coop": "Co-op",
    "cat_early_access": "Acesso antecipado",
    "genre_action": "Gênero Action",
    "genre_adventure": "Gênero Adventure",
    "genre_indie": "Gênero Indie",
    "genre_rpg": "Gênero RPG",
    "genre_strategy": "Gênero Strategy",
    "genre_simulation": "Gênero Simulation",
    "genre_casual": "Gênero Casual",
    "genre_racing": "Gênero Racing",
    "genre_sports": "Gênero Sports",
}


def load_model_and_config():
    """Carrega modelo treinado e configuração salva no treinamento."""
    model_path = MODELS_DIR / "random_forest.joblib"
    config_path = ARTIFACTS_DIR / "model_config.json"

    if not model_path.exists() or not config_path.exists():
        raise FileNotFoundError(
            "Modelo não encontrado. Execute: python -m src.train"
        )

    model = joblib.load(model_path)
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    return model, config


def build_input_dataframe(user_input: dict, feature_columns: list[str]) -> pd.DataFrame:
    """
    Converte o dicionário do formulário Streamlit em DataFrame no formato do treino.

    Inicia todas as colunas em 0 e ativa (1) apenas gêneros/categorias selecionados.
    """
    row = {col: 0 for col in feature_columns}

    row["price_usd"] = float(user_input.get("price_usd", 0))
    row["is_free"] = int(user_input.get("is_free", False))
    row["num_languages"] = int(user_input.get("num_languages", 1))
    row["achievement_count"] = int(user_input.get("achievement_count", 0))
    row["release_year"] = int(user_input.get("release_year", 2024))
    row["release_month"] = int(user_input.get("release_month", 6))
    row["mat_supports_windows"] = int(user_input.get("supports_windows", True))
    row["mat_supports_mac"] = int(user_input.get("supports_mac", False))
    row["mat_supports_linux"] = int(user_input.get("supports_linux", False))
    row["publisher_tier"] = int(user_input.get("publisher_tier", 0))
    row["has_controller"] = int(user_input.get("has_controller", False))
    row["num_categories"] = int(user_input.get("num_categories", 5))

    for genre in user_input.get("genres", []):
        key = f"genre_{genre.lower()}"
        if key in row:
            row[key] = 1

    flags = {
        "cat_multiplayer": user_input.get("multiplayer", False),
        "cat_singleplayer": user_input.get("singleplayer", True),
        "cat_achievements": user_input.get("achievements", False),
        "cat_coop": user_input.get("coop", False),
    }
    for key, value in flags.items():
        if key in row:
            row[key] = int(value)

    return pd.DataFrame([row])[feature_columns].astype(float)


def predict_success(user_input: dict) -> dict:
    """
    Executa predição completa com explicabilidade.

    Retorna: classe, probabilidades, detalhes das features e texto explicativo.
    """
    model, config = load_model_and_config()
    feature_columns = config["feature_columns"]

    X = build_input_dataframe(user_input, feature_columns)
    pred_class = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]

    # Explicabilidade: combina importância global × valor da entrada nesta simulação
    importances = model.feature_importances_
    input_values = X.iloc[0].values
    contribution = importances * np.abs(input_values - input_values.mean())
    top_idx = np.argsort(contribution)[::-1][:8]

    feature_details = []
    for idx in top_idx:
        fname = feature_columns[idx]
        feature_details.append(
            {
                "feature": fname,
                "label": FEATURE_LABELS_PT.get(fname, fname),
                "importance": float(importances[idx]),
                "value": float(input_values[idx]),
                "contribution": float(contribution[idx]),
            }
        )

    explanation = build_text_explanation(user_input, pred_class, feature_details)

    return {
        "class_id": pred_class,
        "class_label": SUCCESS_LABELS[pred_class],
        "probabilities": {
            SUCCESS_LABELS[i]: float(proba[i]) for i in range(len(proba))
        },
        "feature_details": feature_details,
        "explanation": explanation,
    }


def build_text_explanation(user_input: dict, pred_class: int, features: list[dict]) -> str:
    """
    Gera explicação em linguagem natural a partir de regras simples + top features.

    Camada complementar ao gráfico: traduz o resultado para o desenvolvedor indie.
    """
    label = SUCCESS_LABELS[pred_class]
    positives = []
    negatives = []

    if user_input.get("price_usd", 0) <= 9.99 and not user_input.get("is_free"):
        positives.append("preço acessível")
    elif user_input.get("is_free"):
        positives.append("modelo free-to-play")
    elif user_input.get("price_usd", 0) > 29.99:
        negatives.append("preço elevado")

    if user_input.get("multiplayer"):
        positives.append("suporte multiplayer")
    if user_input.get("num_languages", 1) >= 5:
        positives.append("amplo suporte de idiomas")
    if user_input.get("achievements"):
        positives.append("presença de achievements")
    if user_input.get("has_controller"):
        positives.append("suporte a controle")
    if "Indie" in user_input.get("genres", []):
        positives.append("posicionamento indie")

    if user_input.get("publisher_tier", 0) == 0:
        negatives.append("publisher com histórico modesto")
    if user_input.get("num_categories", 0) < 3:
        negatives.append("poucas tags/categorias na página")

    top_feats = [f["label"] for f in features[:3]]
    feat_text = ", ".join(top_feats)

    pos_text = ", ".join(positives) if positives else "poucos indicadores positivos claros"
    neg_text = ", ".join(negatives) if negatives else "nenhum fator negativo dominante"

    return (
        f"O modelo classificou o jogo como **{label}**. "
        f"Os fatores mais influentes foram: {feat_text}. "
        f"Pontos positivos identificados: {pos_text}. "
        f"Pontos de atenção: {neg_text}. "
        f"Esta previsão reflete padrões históricos da Steam e não garante resultados futuros."
    )
