# Champion parameters
CHAMPION_FEATURES = [
    "style",  # 0-100, basic-attack vs ability user
    "damage",  # categorical/step (e.g. 0/1/2)
    "toughness",  # 0/1/2
    "control",  # 0/1/2
    "mobility",  # 0/1/2
    "utility",  # 0/1/2
    "pointclick_cc",  # int: point-and-click CC count
    "skillshot_cc",  # int: skillshot CC count
]

# Model hyperparameters
EMBEDDING_CONFIG = {
    "input_dim": len(CHAMPION_FEATURES),
    "hidden_dims": [64, 32],
    "embedding_dim": 2,  # learned embedding size
    "dropout": 0.2,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 1000,
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

MODEL_DIR = "saved_models"
EMBEDDINGS_PATH = f"{MODEL_DIR}/champion_embeddings.pth"
EMBEDDING_MODEL_PATH = f"{MODEL_DIR}/embedding_model.pth"
PICKER_MODEL_PATH = f"{MODEL_DIR}/picker_model.pth"
