import torch
import numpy as np
from models import ChampionEmbedding, DraftPicker
from utils import load_champion_stats, load_champion_embeddings
from config import *


class DraftPredictor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load embedding model
        self.embedding_model = ChampionEmbedding(
            input_dim=EMBEDDING_CONFIG["input_dim"],
            hidden_dims=EMBEDDING_CONFIG["hidden_dims"],
            embedding_dim=EMBEDDING_CONFIG["embedding_dim"],
            dropout=0,  # No dropout during inference
        )
        self.embedding_model.load_state_dict(
            torch.load(EMBEDDING_MODEL_PATH, map_location=self.device)
        )
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        # Load picker model (role-specific inputs)
        self.picker_model = DraftPicker(
            embedding_dim=EMBEDDING_CONFIG["embedding_dim"],
            hidden_dims=PICKER_CONFIG["hidden_dims"],
            output_dim=EMBEDDING_CONFIG["embedding_dim"],
            dropout=0,  # No dropout during inference
        )
        self.picker_model.load_state_dict(
            torch.load(PICKER_MODEL_PATH, map_location=self.device)
        )
        self.picker_model.to(self.device)
        self.picker_model.eval()
        # Load champion embeddings
        embeddings_path = f"{MODEL_DIR}/champion_embeddings.pth"
        self.champion_embeddings = load_champion_embeddings(embeddings_path)
        self.champion_names = list(self.champion_embeddings.keys())
        # Load champion stats for CC lookups
        self.champion_stats = load_champion_stats(CHAMPION_STATS_FILE)
        # Stack all embeddings for efficient distance computation
        self.all_embeddings = torch.stack(
            [self.champion_embeddings[name] for name in self.champion_names]
        ).to(self.device)

    def _get_embedding_or_zero(self, champion_name):
        if champion_name is None or champion_name == "":
            return torch.zeros(EMBEDDING_CONFIG["embedding_dim"], device=self.device)
        return self.champion_embeddings.get(
            champion_name,
            torch.zeros(EMBEDDING_CONFIG["embedding_dim"], device=self.device),
        )

    def _compute_team_cc(self, champion_names):
        pc_cc_vals = []
        skillshot_cc_vals = []
        for name in champion_names:
            if name and name in self.champion_stats.index:
                pc_cc_vals.append(self.champion_stats.loc[name, "pointclick_cc"])
                skillshot_cc_vals.append(self.champion_stats.loc[name, "skillshot_cc"])
        # Return mean or 0 if no champions
        mean_pc = np.mean(pc_cc_vals) if pc_cc_vals else 0.0
        mean_ss = np.mean(skillshot_cc_vals) if skillshot_cc_vals else 0.0
        return torch.tensor([mean_pc, mean_ss], dtype=torch.float32, device=self.device)

    def predict(
        self,
        enemy_adc=None,
        enemy_support=None,
        my_support=None,
        teammate_1=None,
        teammate_2=None,
        teammate_3=None,
        enemy_1=None,
        enemy_2=None,
        enemy_3=None,
        available_champions=None,
    ):
        with torch.no_grad():
            # Get embeddings for specific roles (zero if not picked)
            enemy_adc_emb = self._get_embedding_or_zero(enemy_adc).unsqueeze(0)
            enemy_support_emb = self._get_embedding_or_zero(enemy_support).unsqueeze(0)
            my_support_emb = self._get_embedding_or_zero(my_support).unsqueeze(0)
            # Compute aggregate CC stats
            my_teammates = [
                name for name in [teammate_1, teammate_2, teammate_3] if name
            ]
            enemy_others = [name for name in [enemy_1, enemy_2, enemy_3] if name]
            my_team_cc = self._compute_team_cc(my_teammates).unsqueeze(0)
            enemy_team_cc = self._compute_team_cc(enemy_others).unsqueeze(0)
            # Predict optimal embedding
            optimal_embedding = self.picker_model(
                enemy_adc_emb,
                enemy_support_emb,
                my_support_emb,
                my_team_cc,
                enemy_team_cc,
            )
            # Find closest champion by embedding distance
            if available_champions:
                # Filter to only available champions
                available_indices = [
                    i
                    for i, name in enumerate(self.champion_names)
                    if name in available_champions
                ]
                candidate_embeddings = self.all_embeddings[available_indices]
                candidate_names = [self.champion_names[i] for i in available_indices]
            else:
                candidate_embeddings = self.all_embeddings
                candidate_names = self.champion_names
            # Compute distances
            distances = torch.norm(candidate_embeddings - optimal_embedding, dim=1)
            best_idx = torch.argmin(distances).item()
            best_champion = candidate_names[best_idx]
            # Compute confidence (inverse of normalized distance)
            min_dist = distances[best_idx].item()
            max_dist = distances.max().item()
            confidence = 1 - (min_dist / (max_dist + 1e-6))
            return best_champion, confidence

    def get_top_k_recommendations(
        self,
        k=5,
        enemy_adc=None,
        enemy_support=None,
        my_support=None,
        teammate_1=None,
        teammate_2=None,
        teammate_3=None,
        enemy_1=None,
        enemy_2=None,
        enemy_3=None,
        available_champions=None,
    ):
        with torch.no_grad():
            # Get embeddings for specific roles (zero if not picked)
            enemy_adc_emb = self._get_embedding_or_zero(enemy_adc).unsqueeze(0)
            enemy_support_emb = self._get_embedding_or_zero(enemy_support).unsqueeze(0)
            my_support_emb = self._get_embedding_or_zero(my_support).unsqueeze(0)
            # Compute aggregate CC stats
            my_teammates = [
                name for name in [teammate_1, teammate_2, teammate_3] if name
            ]
            enemy_others = [name for name in [enemy_1, enemy_2, enemy_3] if name]
            my_team_cc = self._compute_team_cc(my_teammates).unsqueeze(0)
            enemy_team_cc = self._compute_team_cc(enemy_others).unsqueeze(0)
            # Predict optimal embedding
            optimal_embedding = self.picker_model(
                enemy_adc_emb,
                enemy_support_emb,
                my_support_emb,
                my_team_cc,
                enemy_team_cc,
            )
            # Filter candidates
            if available_champions:
                available_indices = [
                    i
                    for i, name in enumerate(self.champion_names)
                    if name in available_champions
                ]
                candidate_embeddings = self.all_embeddings[available_indices]
                candidate_names = [self.champion_names[i] for i in available_indices]
            else:
                candidate_embeddings = self.all_embeddings
                candidate_names = self.champion_names
            # Compute distances and get top K
            distances = torch.norm(candidate_embeddings - optimal_embedding, dim=1)
            top_k_indices = torch.topk(distances, k, largest=False).indices
            # Convert to scores (inverse distance)
            max_dist = distances.max().item()
            recommendations = []
            for idx in top_k_indices:
                champ_name = candidate_names[idx.item()]
                dist = distances[idx].item()
                score = 1 - (dist / (max_dist + 1e-6))
                recommendations.append((champ_name, score))
            return recommendations


def main():
    print("Loading models...")
    predictor = DraftPredictor()
    # Example draft scenario
    enemy_adc = "Caitlyn"
    enemy_support = "Lux"
    my_support = "Thresh"
    teammate_1 = "Malphite"  # Top
    teammate_2 = "Zed"  # Mid
    teammate_3 = None  # Jungle not picked yet
    enemy_1 = "Yasuo"  # Mid
    enemy_2 = None
    enemy_3 = None
    print("\nDraft Scenario:")
    print(f"  Enemy ADC: {enemy_adc}")
    print(f"  Enemy Support: {enemy_support}")
    print(f"  Enemy Others: {', '.join([e for e in [enemy_1, enemy_2, enemy_3] if e])}")
    print(f"  My Support: {my_support}")
    print(
        f"  My Teammates: {', '.join([t for t in [teammate_1, teammate_2, teammate_3] if t])}"
    )
    # Get prediction
    best_pick, confidence = predictor.predict(
        enemy_adc=enemy_adc,
        enemy_support=enemy_support,
        my_support=my_support,
        teammate_1=teammate_1,
        teammate_2=teammate_2,
        teammate_3=teammate_3,
        enemy_1=enemy_1,
        enemy_2=enemy_2,
        enemy_3=enemy_3,
    )
    print(f"\nRecommended Pick: {best_pick}")
    print(f"   Confidence: {confidence:.2%}")
    # Get top 5 recommendations
    print("\nTop 5 Recommendations:")
    top_picks = predictor.get_top_k_recommendations(
        k=5,
        enemy_adc=enemy_adc,
        enemy_support=enemy_support,
        my_support=my_support,
        teammate_1=teammate_1,
        teammate_2=teammate_2,
        teammate_3=teammate_3,
        enemy_1=enemy_1,
        enemy_2=enemy_2,
        enemy_3=enemy_3,
    )
    for i, (champ, score) in enumerate(top_picks, 1):
        print(f"  {i}. {champ} (score: {score:.3f})")


if __name__ == "__main__":
    main()
