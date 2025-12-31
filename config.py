# Champion parameters
CHAMPION_NUMERIC_FEATURES = [
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
CHAMPION_CATEGORICAL_FEATURES = [
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

# Item stats
ITEM_STATS = [
    "attack_damage",
    "attack_speed",
    "ability_power",
    "armor",
    "magic_resistance",
    "lethality",
    "armor_penetration",
    "magic_penetration",
    "health",
    "ability_haste",
    "base_mana_regeneration",
    "heal_and_shield_power",
    "movement_speed",
    "mana",
    "life_steal",
    "critical_strike_chance",
    "base_health_regeneration",
    "omnivamp",
    "critical_strike_damage",
    "tenacity",
]

# Whether to use one-hot encoding for class (True) or label encoding (False)
USE_CLASS_ONEHOT = True
EMBEDDING_INPUT_DIMENSIONS = len(CHAMPION_NUMERIC_FEATURES) + (
    len(CHAMPION_CLASSES) if USE_CLASS_ONEHOT else 1
)

# Model hyperparameters
EMBEDDING_CONFIG = {
    "input_dim": EMBEDDING_INPUT_DIMENSIONS,
    "embedding_dim": 3,
    "hidden_dims": [32],
    "dropout": 0.15,
    "learning_rate": 0.001,
    "batch_size": 8,
    "epochs": 1000,
    # Loss weights
    "lambda_distance": 1.0,
    "lambda_uniformity": 1.0,
    "lambda_ortho": 0.15,
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
DRAFT_HISTORY_FILE = f"{DATA_DIR}/draft_history.json"
CHAMPION_STATS_FILE = f"{DATA_DIR}/champion_stats.csv"
CHAMPION_NAMES_FILE = f"{DATA_DIR}/champion_names.txt"
SCRAP_CHAMPION_DATA_PATH = f"{DATA_DIR}/scraped_champions"
ITEM_STATS_FILE = f"{DATA_DIR}/item_stats.csv"
ITEM_NAMES_FILE = f"{DATA_DIR}/item_names.txt"
SCRAP_ITEM_DATA_PATH = f"{DATA_DIR}/scraped_items"

MODEL_DIR = "saved_models"
EMBEDDINGS_PATH = f"{MODEL_DIR}/champion_embeddings.pth"
EMBEDDINGS_NORMALIZED_PATH = f"{MODEL_DIR}/champion_embeddings_normalized.pth"
EMBEDDING_MODEL_PATH = f"{MODEL_DIR}/embedding_model.pth"
EMBEDDING_NORMALIZER_MODEL_PATH = f"{MODEL_DIR}/embedding_normalizer.pth"
PICKER_MODEL_PATH = f"{MODEL_DIR}/picker_model.pth"
