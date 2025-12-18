import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import os
from config import EMBEDDINGS_NORMALIZED_PATH, EMBEDDINGS_PATH


def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return torch.load(path, map_location="cpu")


def visualize_embeddings(
    embeddings_dict,
    annotate=True,
    figsize=(12, 12),
):
    names = list(embeddings_dict.keys())
    embeddings = torch.stack(
        [
            emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
            for emb in embeddings_dict.values()
        ]
    ).numpy()
    # Dimensionality handling
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        variance = pca.explained_variance_ratio_.sum()
        title = f"Champion Embeddings (PCA → 2D, {variance:.1%} variance)"
    else:
        embeddings_2d = embeddings
        title = "Champion Embeddings (Native 2D)"
    x = embeddings_2d[:, 0]
    y = embeddings_2d[:, 1]
    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(x, y, alpha=0.8)
    if annotate:
        for i, name in enumerate(names):
            plt.text(
                x[i],
                y[i],
                name,
                fontsize=8,
                alpha=0.75,
            )
    # Auto limits with padding
    pad_x = (x.max() - x.min()) * 0.1
    pad_y = (y.max() - y.min()) * 0.1
    plt.xlim(x.min() - pad_x, x.max() + pad_x)
    plt.ylim(y.min() - pad_y, y.max() + pad_y)
    # Labeling
    plt.xlabel("Embedding dim 1")
    plt.ylabel("Embedding dim 2")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Champion Embeddings")
    parser.add_argument(
        "--embeddings",
        type=str,
        # default=EMBEDDINGS_PATH,
        default=EMBEDDINGS_NORMALIZED_PATH,
        help="Path to saved champion embeddings (.pth)",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable champion name labels",
    )
    args = parser.parse_args()
    embeddings = load_embeddings(args.embeddings)
    visualize_embeddings(
        embeddings,
        annotate=not args.no_labels,
    )


if __name__ == "__main__":
    main()
