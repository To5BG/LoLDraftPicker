import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import argparse
import os
from config import *
import numpy as np
import plotly.graph_objects as go


def load_embeddings(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def visualize_embeddings_matplotlib(
    embeddings_dict,
    annotate=True,
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
    plt.figure(figsize=(12, 12))
    plt.scatter(x, y, alpha=0.8, s=100)
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
    # Labels and styling
    plt.xlabel("Embedding dim 1")
    plt.ylabel("Embedding dim 2")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_embeddings_3d(embeddings_dict):
    names = list(embeddings_dict.keys())
    embeddings = torch.stack(
        [
            emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
            for emb in embeddings_dict.values()
        ]
    ).numpy()
    # Handle dimensionality
    if embeddings.shape[1] > 3:
        # Reduce to 3D using PCA
        pca = PCA(n_components=3)
        embeddings_3d = pca.fit_transform(embeddings)
        variance = pca.explained_variance_ratio_.sum()
        title = f"Champion Embeddings - PCA → 3D ({variance:.1%} variance)"
    elif embeddings.shape[1] == 2:
        # Add zero z-dimension for 2D embeddings
        embeddings_3d = np.column_stack([embeddings, np.zeros(len(embeddings))])
        title = "Champion Embeddings - 2D + Z=0"
    else:
        # Native 3D
        embeddings_3d = embeddings
        title = "Champion Embeddings - Native 3D"
    # Extract coordinates
    x = embeddings_3d[:, 0]
    y = embeddings_3d[:, 1]
    z = embeddings_3d[:, 2]
    # Create 3D scatter plot
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers+text",
                marker=dict(
                    size=8,
                    color=z,
                    colorscale="Viridis",
                    showscale=True,
                    opacity=0.8,
                    line=dict(width=0.5, color="white"),
                ),
                text=names,
                textposition="top center",
                textfont=dict(size=8),
                hovertemplate="<b>%{text}</b><br>"
                + "X: %{x:.3f}<br>"
                + "Y: %{y:.3f}<br>"
                + "Z: %{z:.3f}<br>"
                + "<extra></extra>",
            )
        ]
    )
    # Update layout for better 3D visualization
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(
                title="Dimension 1",
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
            ),
            yaxis=dict(
                title="Dimension 2",
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
            ),
            zaxis=dict(
                title="Dimension 3",
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="white",
            ),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=1000,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.show()


if __name__ == "__main__":
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
        help="Disable champion name labels (matplotlib only)",
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        help="Use Plotly for true 3D interactive visualization (instead of Matplotlib 2D)",
    )
    args = parser.parse_args()
    embeddings = load_embeddings(args.embeddings)
    if args.__dict__["3d"]:
        visualize_embeddings_3d(embeddings)
    else:
        visualize_embeddings_matplotlib(embeddings, annotate=not args.no_labels)
