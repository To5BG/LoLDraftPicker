import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
import os

from models import ChampionEmbedding
from train.trainers import EmbeddingTrainer
from utils import ChampionStatsDataset, load_champion_stats, save_champion_embeddings
from utils import EmbeddingNormalizer, save_normalizer
from config import *


def train_embedding_model():
    print("=" * 60)
    print("Training Champion Embedding Model")
    print("=" * 60)

    # Load champion stats
    print(f"\nLoading champion stats from {CHAMPION_STATS_FILE}...")
    champion_df = load_champion_stats(CHAMPION_STATS_FILE)
    print(f"Loaded {len(champion_df)} champions")
    dataset = ChampionStatsDataset(
        champion_df,
        CHAMPION_FEATURES,
        use_class_onehot=USE_CLASS_ONEHOT,
    )
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # Create dataloaders
    # Ensure effective batch size does not exceed dataset size
    effective_batch_size = min(
        EMBEDDING_CONFIG["batch_size"], max(1, len(train_dataset))
    )
    train_loader = DataLoader(
        train_dataset, batch_size=effective_batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
    # Initialize model
    model = ChampionEmbedding(
        input_dim=EMBEDDING_CONFIG["input_dim"],
        hidden_dims=EMBEDDING_CONFIG["hidden_dims"],
        embedding_dim=EMBEDDING_CONFIG["embedding_dim"],
        dropout=EMBEDDING_CONFIG["dropout"],
    )
    # Setup training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    # Train
    trainer = EmbeddingTrainer(model, device)
    optimizer = optim.Adam(model.parameters(), lr=EMBEDDING_CONFIG["learning_rate"])
    # Training loop
    best_val_loss = float("inf")
    print(f"\nTraining for {EMBEDDING_CONFIG['epochs']} epochs...")
    print(
        f"Loss weights: distance={trainer.lambda_distance}, "
        f"uniformity={trainer.lambda_uniformity}, ortho={trainer.lambda_ortho}"
    )
    for epoch in range(EMBEDDING_CONFIG["epochs"]):
        train_loss, train_components = trainer.train_epoch(train_loader, optimizer)
        val_loss, _ = trainer.evaluate(val_loader)
        print(
            f"Epoch {epoch+1}/{EMBEDDING_CONFIG['epochs']} - "
            f"Train Loss: {train_loss:.4f} (dist:{train_components['distance']:.4f}, "
            f"unif:{train_components['uniformity']:.4f}, orth:{train_components['orthogonality']:.4f}) | "
            f"Val Loss: {val_loss:.4f}"
        )
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(model.state_dict(), EMBEDDING_MODEL_PATH)
            print(f"  → Saved new best model (val_loss: {val_loss:.4f})")
    # Generate embeddings for all champions
    print("\nGenerating embeddings for all champions...")
    model.eval()
    embeddings_dict = {}
    embeddings_list = []
    with torch.no_grad():
        for i in range(len(dataset.features)):
            features = dataset.features[i].unsqueeze(0).to(device)
            embedding = model(features).squeeze(0).cpu()
            embeddings_dict[dataset.champion_names[i]] = embedding
            embeddings_list.append(embedding)
    # Save raw (unconstrained) embeddings
    save_champion_embeddings(embeddings_dict, EMBEDDINGS_PATH)
    print(f"Saved raw embeddings to {EMBEDDINGS_PATH}")
    # Fit normalizer and save normalization parameters
    print("\nComputing normalization parameters...")
    all_embeddings = torch.stack(embeddings_list)
    normalizer = EmbeddingNormalizer(
        method="percentile",
        percentile_range=(2, 98),
        target_range=(-1, 1),  # Change to (0, 1) if preferred
    )
    normalizer.fit(all_embeddings)
    save_normalizer(normalizer, EMBEDDING_NORMALIZER_MODEL_PATH)
    print(f"Saved normalizer to {EMBEDDING_NORMALIZER_MODEL_PATH}")
    # Optionally save normalized embeddings too
    normalized_embeddings_dict = {}
    for name, emb in embeddings_dict.items():
        normalized_embeddings_dict[name] = normalizer.transform(
            emb.unsqueeze(0)
        ).squeeze(0)
    save_champion_embeddings(normalized_embeddings_dict, EMBEDDINGS_NORMALIZED_PATH)
    print(f"Saved normalized embeddings to {EMBEDDINGS_NORMALIZED_PATH}")
    print("\nEmbedding training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    return embeddings_dict


if __name__ == "__main__":
    train_embedding_model()
