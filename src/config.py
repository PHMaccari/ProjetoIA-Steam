from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "steam_dataset"
DATA_FILE = DATA_DIR / "steamds.csv"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ROOT_DIR / "models"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Gêneros principais (nomes em genres)
MAIN_GENRE_IDS = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Indie": "Indie",
    "RPG": "RPG",
    "Strategy": "Strategy",
    "Simulation": "Simulation",
    "Casual": "Casual",
    "Racing": "Racing",
    "Sports": "Sports",
}

# Categorias canônicas (substrings em categories)
CATEGORY_IDS = {
    "multiplayer": ("Multi-player", "Online PvP", "PvP", "Cross-Platform Multiplayer"),
    "singleplayer": ("Single-player",),
    "achievements": ("Steam Achievements",),
    "controller_full": ("Full controller support",),
    "controller_partial": ("Partial Controller Support",),
    "coop": ("Co-op", "Online Co-op"),
}

MIN_RELEASE_YEAR = 2010
SAMPLE_SIZE = 30_000  # amostra estratificada para treino rápido

SUCCESS_LABELS = {
    0: "Baixo potencial",
    1: "Médio potencial",
    2: "Alto potencial",
}

SUCCESS_COLORS = {
    0: "#ef4444",
    1: "#f59e0b",
    2: "#22c55e",
}
