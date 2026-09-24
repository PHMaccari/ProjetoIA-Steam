"""
Interface Streamlit — Perfil de engajamento e popularidade na Steam.

Telas:
  1. Dashboard  — dataset, classificação e regressão
  2. Simulação  — características conhecidas antes do lançamento
  3. Resultado  — classe de engajamento, recomendações estimadas e explicação
"""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import ARTIFACTS_DIR, DATA_FILE, MAIN_GENRE_IDS, SUCCESS_COLORS, SUCCESS_LABELS
from src.predict import predict_success

st.set_page_config(
    page_title="Steam — Engajamento e Popularidade",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #64748b; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

CLS_NAMES = {
    "logistic_regression": "Regressão Logística (A)",
    "knn": "KNN (B)",
    "random_forest": "Random Forest",
}
REG_NAMES = {
    "dummy_median": "Baseline (mediana)",
    "linear_simple": "Linear simples",
    "linear_multiple": "Linear múltipla",
    "polynomial": "Polinomial + Ridge",
    "random_forest_regressor": "Random Forest Regressor",
}


@st.cache_data(show_spinner="Carregando dataset...")
def load_dashboard_data() -> pd.DataFrame:
    parquet = ARTIFACTS_DIR / "training_dataset.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    from src.data import build_training_dataset

    df = build_training_dataset()
    df.to_parquet(parquet, index=False)
    return df


@st.cache_data
def load_model_metrics() -> dict:
    config_path = ARTIFACTS_DIR / "model_config.json"
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            return json.load(f).get("metrics", {})
    return {}


def _pie_color_map() -> dict[str, str]:
    return {SUCCESS_LABELS[k]: SUCCESS_COLORS[k] for k in SUCCESS_LABELS}


def render_dashboard(df: pd.DataFrame, metrics: dict) -> None:
    st.markdown('<p class="main-header">Dashboard — Engajamento e popularidade</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Classificação do perfil de engajamento e regressão da quantidade de recomendações</p>',
        unsafe_allow_html=True,
    )

    cls_metrics = metrics.get("classification", {})
    reg_metrics = metrics.get("regression", {})
    best_cls = cls_metrics.get("best_model", "random_forest")
    best_reg = reg_metrics.get("best_model", "random_forest_regressor")
    best_cls_acc = cls_metrics.get(best_cls, {}).get("accuracy", metrics.get("accuracy", 0))
    best_reg_mae = reg_metrics.get(best_reg, {}).get("mae")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jogos no dataset", f"{len(df):,}")
    c2.metric("Preço médio (USD)", f"${df['price_usd'].mean():.2f}")
    c3.metric(
        "Acurácia (melhor classificador)",
        f"{best_cls_acc:.1%}",
        help=CLS_NAMES.get(best_cls, best_cls),
    )
    if best_reg_mae is not None:
        c4.metric("MAE da regressão", f"{best_reg_mae:,.0f} rec.")
    else:
        c4.metric("Modelo", "Não treinado")

    class_col = "engagement_class" if "engagement_class" in df.columns else "success_class"

    col_a, col_b = st.columns(2)
    with col_a:
        genre_cols = [c for c in df.columns if c.startswith("genre_")]
        genre_counts = {c.replace("genre_", "").title(): int(df[c].sum()) for c in genre_cols}
        fig_genres = px.bar(
            x=list(genre_counts.keys()),
            y=list(genre_counts.values()),
            labels={"x": "Gênero", "y": "Quantidade"},
            title="Jogos por gênero principal",
            color=list(genre_counts.values()),
            color_continuous_scale="Blues",
        )
        fig_genres.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_genres, width="stretch")

    with col_b:
        success_counts = df[class_col].map(SUCCESS_LABELS).value_counts().reset_index()
        success_counts.columns = ["Classe", "Quantidade"]
        fig_success = px.pie(
            success_counts,
            names="Classe",
            values="Quantidade",
            title="Distribuição do perfil de engajamento",
            color="Classe",
            color_discrete_map=_pie_color_map(),
        )
        st.plotly_chart(fig_success, width="stretch")

    col_c, col_d = st.columns(2)
    with col_c:
        fig_price = px.histogram(
            df,
            x="price_usd",
            nbins=40,
            title="Distribuição de preços (USD)",
            labels={"price_usd": "Preço"},
        )
        st.plotly_chart(fig_price, width="stretch")

    with col_d:
        fig_rec = px.histogram(
            df,
            x="recommendations_total",
            nbins=40,
            title="Distribuição de recomendações (alvo da regressão)",
            labels={"recommendations_total": "Recomendações"},
        )
        st.plotly_chart(fig_rec, width="stretch")

    st.subheader("Comparação dos modelos")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.caption("Classificação (conjunto de teste)")
        if cls_metrics:
            rows = []
            for key, label in CLS_NAMES.items():
                item = cls_metrics.get(key, {})
                if not item:
                    continue
                rows.append(
                    {
                        "Modelo": label,
                        "Accuracy": item.get("accuracy"),
                        "F1 macro": item.get("f1_macro"),
                    }
                )
            table = pd.DataFrame(rows)
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "Accuracy": st.column_config.NumberColumn(format="%.3f"),
                    "F1 macro": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(f"Melhor classificador: **{CLS_NAMES.get(best_cls, best_cls)}**")
        else:
            st.info("Treine o pipeline para ver as métricas de classificação.")

    with col_m2:
        st.caption("Regressão de recomendações (escala original)")
        if reg_metrics:
            rows = []
            for key, label in REG_NAMES.items():
                item = reg_metrics.get(key, {})
                if not isinstance(item, dict) or "mae" not in item:
                    continue
                rows.append(
                    {
                        "Modelo": label,
                        "MAE": item.get("mae"),
                        "RMSE": item.get("rmse"),
                        "R²": item.get("r2"),
                    }
                )
            table = pd.DataFrame(rows)
            st.dataframe(
                table,
                hide_index=True,
                width="stretch",
                column_config={
                    "MAE": st.column_config.NumberColumn(format="%.0f"),
                    "RMSE": st.column_config.NumberColumn(format="%.0f"),
                    "R²": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            st.caption(f"Melhor regressor (menor MAE): **{REG_NAMES.get(best_reg, best_reg)}**")
        else:
            st.info("Treine o pipeline para ver as métricas de regressão.")


def render_simulation() -> dict | None:
    st.markdown('<p class="main-header">Simulação — Engajamento e popularidade</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Informe as características do jogo (conhecidas antes do lançamento)</p>',
        unsafe_allow_html=True,
    )

    with st.form("simulation_form"):
        col1, col2 = st.columns(2)

        with col1:
            is_free = st.checkbox("Jogo gratuito (Free-to-play)", value=False)
            price = st.number_input(
                "Preço (USD)",
                min_value=0.0,
                max_value=99.99,
                value=9.99,
                step=0.99,
                disabled=is_free,
            )
            genres = st.multiselect(
                "Gêneros",
                options=list(MAIN_GENRE_IDS.values()),
                default=["Indie"],
            )
            num_languages = st.slider("Quantidade de idiomas", 1, 20, 3)
            achievement_count = st.number_input("Número de achievements", 0, 500, 10)
            num_categories = st.slider("Quantidade de tags/categorias", 1, 30, 8)

        with col2:
            release_year = st.number_input("Ano de lançamento", 2010, 2026, 2024)
            release_month = st.slider("Mês de lançamento", 1, 12, 6)
            publisher_tier = st.selectbox(
                "Tier do publisher (pelo tamanho do catálogo)",
                options=[0, 1, 2],
                format_func=lambda x: ["Pequeno", "Médio", "Grande"][x],
            )
            multiplayer = st.checkbox("Multiplayer", value=False)
            singleplayer = st.checkbox("Single-player", value=True)
            coop = st.checkbox("Co-op", value=False)
            achievements = st.checkbox("Achievements Steam", value=True)
            has_controller = st.checkbox("Suporte a controle", value=False)
            supports_windows = st.checkbox("Windows", value=True)
            supports_mac = st.checkbox("Mac", value=False)
            supports_linux = st.checkbox("Linux", value=False)

        submitted = st.form_submit_button(
            "Prever engajamento e popularidade",
            type="primary",
            width="stretch",
        )

    if submitted:
        return {
            "price_usd": 0.0 if is_free else price,
            "is_free": is_free,
            "genres": genres,
            "num_languages": num_languages,
            "achievement_count": achievement_count,
            "num_categories": num_categories,
            "release_year": release_year,
            "release_month": release_month,
            "publisher_tier": publisher_tier,
            "multiplayer": multiplayer,
            "singleplayer": singleplayer,
            "coop": coop,
            "achievements": achievements,
            "has_controller": has_controller,
            "supports_windows": supports_windows,
            "supports_mac": supports_mac,
            "supports_linux": supports_linux,
        }
    return None


def render_result(result: dict) -> None:
    st.markdown('<p class="main-header">Resultado da previsão</p>', unsafe_allow_html=True)

    class_id = result["class_id"]
    color = SUCCESS_COLORS[class_id]
    rec_pred = result.get("recommendations_pred")

    st.markdown(
        f"""
        <div style="background:{color}22;border-left:6px solid {color};
        padding:1.2rem;border-radius:8px;margin-bottom:1.5rem;">
        <h2 style="color:{color};margin:0;">{result['class_label']}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if rec_pred is not None:
        st.metric("Popularidade estimada", f"{rec_pred:,.0f} recomendações")

    prob_df = pd.DataFrame(
        {
            "Classe": list(result["probabilities"].keys()),
            "Probabilidade": list(result["probabilities"].values()),
        }
    )

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.subheader("Probabilidades do classificador")
        for label, prob in result["probabilities"].items():
            st.progress(prob, text=f"{label}: {prob:.1%}")

        st.subheader("Explicação")
        st.markdown(result["explanation"])

    with col2:
        st.subheader("Importância das variáveis")
        feat_df = pd.DataFrame(result["feature_details"])
        fig = px.bar(
            feat_df,
            x="contribution",
            y="label",
            orientation="h",
            title="Fatores que mais influenciaram a classificação",
            labels={"contribution": "Contribuição", "label": "Variável"},
            color="contribution",
            color_continuous_scale="Viridis",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

        fig_prob = px.bar(
            prob_df,
            x="Classe",
            y="Probabilidade",
            color="Classe",
            title="Distribuição de probabilidades",
            color_discrete_map=_pie_color_map(),
        )
        st.plotly_chart(fig_prob, width="stretch")


def main() -> None:
    st.sidebar.title("Steam Engajamento")
    st.sidebar.markdown("**Machine Learning**")
    st.sidebar.markdown(
        "Classificação do perfil de engajamento e regressão da popularidade (recomendações)."
    )

    page = st.sidebar.radio(
        "Navegação",
        ["Dashboard", "Simulação", "Resultado"],
        index=0,
    )

    if not (ARTIFACTS_DIR / "model_config.json").exists():
        st.sidebar.warning("Modelo não treinado. Execute: `python -m src.train`")

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    if page == "Dashboard":
        try:
            df = load_dashboard_data()
            metrics = load_model_metrics()
            render_dashboard(df, metrics)
        except FileNotFoundError as e:
            st.error(f"Dados não encontrados: {e}")
            st.info(f"Coloque o arquivo em: `{DATA_FILE}`")

    elif page == "Simulação":
        user_input = render_simulation()
        if user_input:
            try:
                st.session_state.prediction_result = predict_success(user_input)
                st.success("Previsão concluída! Veja a aba **Resultado**.")
            except FileNotFoundError:
                st.error("Modelo não encontrado. Execute primeiro: `python -m src.train`")

    elif page == "Resultado":
        if st.session_state.prediction_result:
            render_result(st.session_state.prediction_result)
        else:
            st.info("Faça uma simulação primeiro na aba **Simulação**.")


if __name__ == "__main__":
    main()
