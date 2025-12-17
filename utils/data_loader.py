import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ChampionStatsDataset(Dataset):
    def __init__(self, champion_stats_df, feature_columns):
        self.features = torch.FloatTensor(champion_stats_df[feature_columns].values)
        # Infer target embedding dim from config, or default 2
        try:
            from config import EMBEDDING_CONFIG

            target_dim = EMBEDDING_CONFIG.get("embedding_dim", 2)
        except Exception:
            target_dim = 2
        # Init
        self.targets = torch.FloatTensor(
            np.random.randn(len(champion_stats_df), target_dim)
        )
        self.champion_names = champion_stats_df["champion_name"].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


class DraftDataset(Dataset):
    def __init__(self, draft_data, champion_embeddings_dict, champion_stats_df):
        self.draft_data = draft_data
        self.champion_embeddings_dict = champion_embeddings_dict
        self.embedding_dim = next(iter(champion_embeddings_dict.values())).shape[0]
        # Create lookup for CC stats
        self.cc_stats = {}
        for _, row in champion_stats_df.iterrows():
            self.cc_stats[row["champion_name"]] = {
                "pointclick_cc": row["pointclick_cc"],
                "skillshot_cc": row["skillshot_cc"],
            }

    def __len__(self):
        return len(self.draft_data)

    def __getitem__(self, idx):
        scenario = self.draft_data[idx]
        # Get specific role embeddings (or zeros if not picked)
        enemy_adc_emb = (
            self.champion_embeddings_dict.get(scenario.get("enemy_adc"))
            if scenario.get("enemy_adc")
            else torch.zeros(self.embedding_dim)
        )
        enemy_support_emb = (
            self.champion_embeddings_dict.get(scenario.get("enemy_support"))
            if scenario.get("enemy_support")
            else torch.zeros(self.embedding_dim)
        )
        my_support_emb = (
            self.champion_embeddings_dict.get(scenario.get("my_support"))
            if scenario.get("my_support")
            else torch.zeros(self.embedding_dim)
        )
        # Aggregate CC stats for my team
        my_team_champs = [
            scenario.get(f"teammate_{i}")
            for i in range(1, 4)
            if scenario.get(f"teammate_{i}")
        ]
        if my_team_champs:
            my_team_pointclick = sum(
                self.cc_stats[c]["pointclick_cc"] for c in my_team_champs
            ) / len(my_team_champs)
            my_team_skillshot = sum(
                self.cc_stats[c]["skillshot_cc"] for c in my_team_champs
            ) / len(my_team_champs)
        else:
            my_team_pointclick = 0.0
            my_team_skillshot = 0.0
        my_team_cc = torch.tensor(
            [my_team_pointclick, my_team_skillshot], dtype=torch.float
        )
        # Aggregate CC stats for enemy team
        enemy_team_champs = [
            scenario.get(f"enemy_{i}")
            for i in range(1, 4)
            if scenario.get(f"enemy_{i}")
        ]
        if enemy_team_champs:
            enemy_team_pointclick = sum(
                self.cc_stats[c]["pointclick_cc"] for c in enemy_team_champs
            ) / len(enemy_team_champs)
            enemy_team_skillshot = sum(
                self.cc_stats[c]["skillshot_cc"] for c in enemy_team_champs
            ) / len(enemy_team_champs)
        else:
            enemy_team_pointclick = 0.0
            enemy_team_skillshot = 0.0
        enemy_team_cc = torch.tensor(
            [enemy_team_pointclick, enemy_team_skillshot], dtype=torch.float
        )
        target_emb = self.champion_embeddings_dict[scenario["optimal_pick"]]
        return (
            enemy_adc_emb,
            enemy_support_emb,
            my_support_emb,
            my_team_cc,
            enemy_team_cc,
            target_emb,
        )


def load_champion_stats(filepath):
    return pd.read_csv(filepath)


def load_draft_history(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


def save_champion_embeddings(embeddings_dict, filepath):
    torch.save(embeddings_dict, filepath)


def load_champion_embeddings(filepath):
    return torch.load(filepath)
