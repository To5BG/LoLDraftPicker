import torch
from utils import load_champion_embeddings, load_normalizer
from config import EMBEDDINGS_PATH, EMBEDDING_NORMALIZER_MODEL_PATH


class ChampionEmbeddingQuery:
    def __init__(
        self,
        embeddings_path=EMBEDDINGS_PATH,
        normalizer_path=EMBEDDING_NORMALIZER_MODEL_PATH,
    ):
        print(f"Loading embeddings from {embeddings_path}...")
        self.embeddings_raw = load_champion_embeddings(embeddings_path)
        print(f"Loading normalizer from {normalizer_path}...")
        self.normalizer = load_normalizer(normalizer_path)
        self.champion_names = list(self.embeddings_raw.keys())
        print(f"Loaded {len(self.champion_names)} champions")

    def get_embedding(self, champion_name, normalized=True):
        # Case-insensitive lookup
        champion_name = champion_name.lower()
        matching = [
            name for name in self.champion_names if name.lower() == champion_name
        ]
        if not matching:
            raise ValueError(f"Champion '{champion_name}' not found.")
        emb = self.embeddings_raw[matching[0]]
        if normalized:
            emb = self.normalizer.transform(emb.unsqueeze(0)).squeeze(0)
        return emb

    def get_all_embeddings(self, normalized=True, as_dict=False):
        if normalized:
            embeddings = {
                name: self.normalizer.transform(emb.unsqueeze(0)).squeeze(0)
                for name, emb in self.embeddings_raw.items()
            }
        else:
            embeddings = self.embeddings_raw
        if as_dict:
            return embeddings
        else:
            names = list(embeddings.keys())
            embs = torch.stack([embeddings[name] for name in names])
            return names, embs

    def find_similar(self, champion_name, top_k=5, normalized=True):
        query_emb = self.get_embedding(champion_name, normalized=normalized)
        names, all_embs = self.get_all_embeddings(normalized=normalized, as_dict=False)
        # Compute distances
        distances = torch.norm(all_embs - query_emb.unsqueeze(0), dim=1)
        # Sort and get top-k (excluding self at index 0)
        sorted_indices = torch.argsort(distances)
        results = []
        for idx in sorted_indices:
            name = names[idx]
            if name.lower() != champion_name.lower():
                results.append((name, distances[idx].item()))
            if len(results) >= top_k:
                break
        return results


def main():
    query = ChampionEmbeddingQuery()
    print("\n" + "=" * 60)
    print("Example: Jinx embedding")
    print("=" * 60)
    # Raw embedding
    jinx_raw = query.get_embedding("jinx", normalized=False)
    print(f"\nJinx (raw): {jinx_raw.numpy()}")
    # Normalized embedding
    jinx_norm = query.get_embedding("jinx", normalized=True)
    print(f"Jinx (normalized): {jinx_norm.numpy()}")
    print("\n" + "=" * 60)
    print("Example: Similar champions to Jinx")
    print("=" * 60)
    similar = query.find_similar("jinx", top_k=10, normalized=True)
    for i, (name, dist) in enumerate(similar, 1):
        print(f"{i}. {name:20s} (distance: {dist:.4f})")


if __name__ == "__main__":
    main()
