import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import os
from config import EMBEDDINGS_PATH


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
    # Axis centering
    x_mid = (x.min() + x.max()) / 2
    y_mid = (y.min() + y.max()) / 2
    x_shift = 0.5 - x_mid
    y_shift = 0.5 - y_mid
    x_vis = x + x_shift
    y_vis = y + y_shift
    # Plot
    plt.figure(figsize=figsize)
    plt.scatter(x_vis, y_vis, alpha=0.8)
    if annotate:
        for i, name in enumerate(names):
            plt.text(
                x_vis[i],
                y_vis[i],
                name,
                fontsize=8,
                alpha=0.75,
            )
    # Draw center lines at (0.5, 0.5)
    plt.axhline(0.5, linewidth=0.5, color="gray", alpha=0.6)
    plt.axvline(0.5, linewidth=0.5, color="gray", alpha=0.6)
    # Auto limits with padding
    pad_x = (x_vis.max() - x_vis.min()) * 0.1
    pad_y = (y_vis.max() - y_vis.min()) * 0.1
    plt.xlim(x_vis.min() - pad_x, x_vis.max() + pad_x)
    plt.ylim(y_vis.min() - pad_y, y_vis.max() + pad_y)
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
        default=EMBEDDINGS_PATH,
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
