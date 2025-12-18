# Champion parameters
CHAMPION_FEATURES = [
    "hp",
    "mp",
    "ar",
    "ad",
    "mr",
    "ms",
    "range",
    "style",
    "damage",
    "toughness",
    "control",
    "mobility",
    "utility",
]

# Model hyperparameters
EMBEDDING_CONFIG = {
    "input_dim": len(CHAMPION_FEATURES),
    "base_margin": 0.13,
    "embedding_dim": 2,
    "hidden_dims": [64, 32],
    "dropout": 0.1,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 100,
}

# Picker config (embedding size is in EMBEDDING_CONFIG)
# Defaults are provided but all code supports variable-length team/enemy groups.
PICKER_CONFIG = {
    "num_team_slots": 4,
    "num_enemy_visible": 2,
    "hidden_dims": [128, 64, 32],
    "dropout": 0.3,
    "learning_rate": 0.0005,
    "batch_size": 8,
    "epochs": 150,
}

# Data and model paths
DATA_DIR = "data"
CHAMPION_STATS_FILE = f"{DATA_DIR}/champion_stats.csv"
DRAFT_HISTORY_FILE = f"{DATA_DIR}/draft_history.json"
SCRAP_DATA_PATH = f"{DATA_DIR}/scraped_champions"

MODEL_DIR = "saved_models"
EMBEDDINGS_PATH = f"{MODEL_DIR}/champion_embeddings.pth"
EMBEDDINGS_NORMALIZED_PATH = f"{MODEL_DIR}/champion_embeddings_normalized.pth"
EMBEDDING_MODEL_PATH = f"{MODEL_DIR}/embedding_model.pth"
EMBEDDING_NORMALIZER_MODEL_PATH = f"{MODEL_DIR}/embedding_normalizer.pth"
PICKER_MODEL_PATH = f"{MODEL_DIR}/picker_model.pth"
