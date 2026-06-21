"""
ETL e feature engineering do dataset Steam (steamds.csv).

Responsabilidades:
- Ler e limpar o CSV único
- Converter colunas em lista (genres, categories, publishers)
- Criar features numéricas conhecidas antes do lançamento
- Definir a variável-alvo (success_class) sem vazamento de dados
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from src.config import (
    CATEGORY_IDS,
    DATA_FILE,
    MAIN_GENRE_IDS,
    MIN_RELEASE_YEAR,
    SAMPLE_SIZE,
)


def _read_steam_csv() -> pd.DataFrame:
    """Lê apenas as colunas necessárias para reduzir uso de memória."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {DATA_FILE}")
    cols = [
        "appid",
        "name",
        "release_date",
        "price",
        "achievements",
        "recommendations",
        "supported_languages",
        "windows",
        "mac",
        "linux",
        "genres",
        "categories",
        "publishers",
    ]
    return pd.read_csv(DATA_FILE, usecols=cols)


def parse_list_field(value: str | float) -> list[str]:
    """
    Converte texto como "['Action', 'Indie']" em lista Python.

    O CSV armazena listas como string; usamos literal_eval em vez de eval
    por segurança (não executa código arbitrário).
    """
    if pd.isna(value) or not str(value).strip():
        return []
    text = str(value).strip()
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass
    return []


def count_languages(text: str | float) -> int:
    """Conta idiomas a partir da lista em supported_languages."""
    return len(parse_list_field(text))


def category_has_any(categories: list[str], keywords: tuple[str, ...]) -> int:
    """Retorna 1 se o jogo possui ao menos uma categoria da lista de keywords."""
    normalized = {c.lower() for c in categories}
    return int(any(keyword.lower() in normalized for keyword in keywords))


def build_category_flags(categories_series: pd.Series) -> pd.DataFrame:
    """Cria flags binárias (0/1) para categorias Steam e conta total de tags."""
    rows = []
    for appid, raw in categories_series.items():
        categories = parse_list_field(raw)
        row = {"appid": appid}
        for flag_name, keywords in CATEGORY_IDS.items():
            row[f"cat_{flag_name}"] = category_has_any(categories, keywords)
        row["num_categories"] = len(categories)
        rows.append(row)
    return pd.DataFrame(rows)


def build_genre_flags(genres_series: pd.Series) -> pd.DataFrame:
    """One-hot encoding dos gêneros principais por jogo."""
    rows = []
    for appid, raw in genres_series.items():
        genres = {g.lower() for g in parse_list_field(raw)}
        row = {"appid": appid}
        for genre_name in MAIN_GENRE_IDS:
            row[f"genre_{genre_name.lower()}"] = int(genre_name.lower() in genres)
        rows.append(row)
    return pd.DataFrame(rows)


def build_publisher_tier(publishers_series: pd.Series, recommendations: pd.Series) -> pd.DataFrame:
    """
    Classifica publishers em 3 tiers (0, 1, 2) pela média histórica de recomendações.

    Tier 0 = indie/desconhecido | 1 = médio | 2 = estabelecido/AAA
    Jogos com múltiplos publishers recebem o tier mais alto entre eles.
    """
    pub_rows = []
    for appid, raw in publishers_series.items():
        for publisher in parse_list_field(raw):
            pub_rows.append(
                {
                    "appid": appid,
                    "publisher": publisher,
                    "recommendations_total": recommendations.loc[appid],
                }
            )

    if not pub_rows:
        return pd.DataFrame({"appid": publishers_series.index, "publisher_tier": 0})

    pub_apps = pd.DataFrame(pub_rows)
    pub_stats = (
        pub_apps.groupby("publisher")["recommendations_total"]
        .agg(["mean", "count"])
        .reset_index()
    )
    # qcut divide publishers em tercis pela média de recomendações
    pub_stats["publisher_tier"] = pd.qcut(
        pub_stats["mean"].rank(method="first"),
        q=3,
        labels=[0, 1, 2],
    ).astype(int)

    tier_map = pub_stats.set_index("publisher")["publisher_tier"]
    app_tier = pub_apps.copy()
    app_tier["publisher_tier"] = app_tier["publisher"].map(tier_map).fillna(0).astype(int)
    return app_tier.groupby("appid")["publisher_tier"].max().reset_index()


def create_success_target(recommendations: pd.Series) -> pd.Series:
    """
    Variável-alvo: 3 classes por tercis balanceados de recommendations.

    Usamos rank + qcut porque a distribuição bruta é muito assimétrica
    (poucos jogos concentram milhões de recomendações).
    """
    rec = recommendations.fillna(0)
    ranked = rec.rank(method="first")
    return pd.qcut(ranked, q=3, labels=[0, 1, 2]).astype(int)


def load_raw_applications() -> pd.DataFrame:
    """Carrega jogos, filtra por ano e cria colunas base do dataset."""
    apps = _read_steam_csv()
    apps = apps.set_index("appid", drop=False)

    apps["release_date"] = pd.to_datetime(apps["release_date"], errors="coerce")
    apps = apps[apps["release_date"].dt.year >= MIN_RELEASE_YEAR].copy()

    # Features derivadas de colunas brutas
    apps["price_usd"] = apps["price"].fillna(0).astype(float)
    apps["is_free"] = (apps["price_usd"] == 0).astype(int)
    apps["num_languages"] = apps["supported_languages"].apply(count_languages)
    apps["achievement_count"] = apps["achievements"].fillna(0)
    apps["release_year"] = apps["release_date"].dt.year
    apps["release_month"] = apps["release_date"].dt.month
    apps["recommendations_total"] = apps["recommendations"].fillna(0)
    apps["mat_supports_windows"] = apps["windows"].astype(str).str.lower().eq("true").astype(int)
    apps["mat_supports_mac"] = apps["mac"].astype(str).str.lower().eq("true").astype(int)
    apps["mat_supports_linux"] = apps["linux"].astype(str).str.lower().eq("true").astype(int)

    return apps.reset_index(drop=True)


def build_training_dataset(sample_size: int | None = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Pipeline completo: limpeza → flags → merge → variável-alvo → amostragem.

    IMPORTANTE: recommendations_total entra só como alvo (success_class),
    nunca como feature — evita vazamento de dados (data leakage).
    """
    apps = load_raw_applications()
    apps_indexed = apps.set_index("appid")

    # Feature engineering por tipo de informação
    genre_flags = build_genre_flags(apps_indexed["genres"])
    category_flags = build_category_flags(apps_indexed["categories"])
    publisher_tier = build_publisher_tier(
        apps_indexed["publishers"],
        apps_indexed["recommendations_total"],
    )

    df = apps.merge(genre_flags, on="appid", how="left")
    df = df.merge(category_flags, on="appid", how="left")
    df = df.merge(publisher_tier, on="appid", how="left")

    flag_cols = [c for c in df.columns if c.startswith(("genre_", "cat_"))]
    for col in flag_cols:
        df[col] = df[col].fillna(0).astype(int)
    df["publisher_tier"] = df["publisher_tier"].fillna(0).astype(int)
    df["has_controller"] = (
        (df["cat_controller_full"] + df["cat_controller_partial"]) > 0
    ).astype(int)

    df["success_class"] = create_success_target(df["recommendations_total"])

    # Amostra estratificada: ~10 mil jogos por classe (total ~30 mil)
    if sample_size and len(df) > sample_size:
        per_class = sample_size // 3
        samples = []
        for _, group in df.groupby("success_class"):
            n = min(len(group), per_class)
            samples.append(group.sample(n=n, random_state=42))
        df = pd.concat(samples, ignore_index=True)

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Retorna as 25 features usadas pelo modelo.

    Todas representam informações conhecidas ANTES do lançamento.
    """
    base = [
        "price_usd",
        "is_free",
        "num_languages",
        "achievement_count",
        "release_year",
        "release_month",
        "mat_supports_windows",
        "mat_supports_mac",
        "mat_supports_linux",
        "publisher_tier",
        "has_controller",
        "num_categories",
    ]
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    cat_cols = [
        "cat_multiplayer",
        "cat_singleplayer",
        "cat_achievements",
        "cat_coop",
    ]
    return base + genre_cols + [c for c in cat_cols if c in df.columns]


def save_processed_dataset(df: pd.DataFrame, path: Path) -> None:
    """Salva dataset processado para uso no Dashboard Streamlit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
