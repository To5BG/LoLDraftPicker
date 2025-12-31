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
    "class",
]

# Champion classes (categorical feature)
CHAMPION_CLASSES = [
    "juggernaut",
    "burst",
    "assassin",
    "marksman",
    "vanguard",
    "battlemage",
    "specialist",
    "catcher",
    "skirmisher",
    "warden",
    "artillery",
    "diver",
    "enchanter",
]

# Whether to use one-hot encoding for class (True) or label encoding (False)
USE_CLASS_ONEHOT = True

# Model hyperparameters
EMBEDDING_CONFIG = {
    "input_dim": len(CHAMPION_FEATURES),
    "embedding_dim": 2,
    "hidden_dims": [],
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 100,
    # Loss weights
    "lambda_distance": 1.0,
    "lambda_uniformity": 1.0,
    "lambda_ortho": 0.1,
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
    "epochs": 200,
}

# Data and model paths
DATA_DIR = "data"
CHAMPION_STATS_FILE = f"{DATA_DIR}/champion_stats.csv"
CHAMPION_NAMES_FILE = f"{DATA_DIR}/champion_names.txt"
DRAFT_HISTORY_FILE = f"{DATA_DIR}/draft_history.json"
SCRAP_DATA_PATH = f"{DATA_DIR}/scraped_champions"

MODEL_DIR = "saved_models"
EMBEDDINGS_PATH = f"{MODEL_DIR}/champion_embeddings.pth"
EMBEDDINGS_NORMALIZED_PATH = f"{MODEL_DIR}/champion_embeddings_normalized.pth"
EMBEDDING_MODEL_PATH = f"{MODEL_DIR}/embedding_model.pth"
EMBEDDING_NORMALIZER_MODEL_PATH = f"{MODEL_DIR}/embedding_normalizer.pth"
PICKER_MODEL_PATH = f"{MODEL_DIR}/picker_model.pth"
