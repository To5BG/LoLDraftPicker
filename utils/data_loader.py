import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder


class ChampionStatsDataset(Dataset):
    def __init__(
        self,
        champion_stats_df,
        numeric_features,
        categorical_features,
        use_class_onehot=True,
        k_pos=3,
        m_neg=20,
    ):
        # Handle numeric features and normalize
        numeric_features = champion_stats_df[numeric_features].values.astype(np.float32)
        numeric_features = StandardScaler().fit_transform(numeric_features)
        # Handle categorical features
        for cat in categorical_features:
            if cat != "class":
                raise ValueError(f"Unsupported categorical feature: {cat}")
            cat_values = champion_stats_df[cat].values
            if use_class_onehot:
                # One-hot encoding
                class_features = (
                    OneHotEncoder()
                    .fit_transform(cat_values.reshape(-1, 1))
                    .astype(np.float32)
                    .toarray()
                )
            else:
                # Label encoding
                class_features = (
                    LabelEncoder()
                    .fit_transform(cat_values)
                    .reshape(-1, 1)
                    .astype(np.float32)
                    .toarray()
                )
        # Concat
        self.features = torch.FloatTensor(
            np.concatenate([numeric_features, class_features], axis=1)
        )
        self.champion_names = champion_stats_df["champion_name"].values
        self.n = len(self.features)
        # Compute pairwise Euclidean distances between all champions
        dists = pairwise_distances(self.features, self.features, metric="euclidean")
        # For each champion, find k_pos closest and m_neg farthest champions
        self.pos_indices = [np.argsort(dists[i])[1 : k_pos + 1] for i in range(self.n)]
        self.neg_indices = [np.argsort(dists[i])[-m_neg:] for i in range(self.n)]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        anchor = self.features[idx]
        # Sample positive
        pos_idx = np.random.choice(self.pos_indices[idx])
        positive = self.features[pos_idx]
        # Sample negative
        neg_idx = np.random.choice(self.neg_indices[idx])
        negative = self.features[neg_idx]
        return anchor, positive, negative


class DraftDataset(Dataset):
    def __init__(self, draft_data, champion_embeddings_dict, champion_stats_df):
        self.draft_data = draft_data
        self.champion_embeddings_dict = champion_embeddings_dict
        # Get embedding dimension from any champion's embedding
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
        # Extract enemy role embeddings, with zero vector placeholder
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
        # Target: the optimal champion embedding to pick in this scenario
        target_emb = self.champion_embeddings_dict[scenario["optimal_pick"]]
        # Return all context features that inform draft decision
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


def load_champion_embeddings(filepath):
    return torch.load(filepath)
