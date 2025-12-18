import torch
import torch.nn.functional as F
from config import EMBEDDING_CONFIG


class EmbeddingTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        # Loss weights (will be set from config)
        self.lambda_distance = EMBEDDING_CONFIG.get("lambda_distance", 1.0)
        self.lambda_uniformity = EMBEDDING_CONFIG.get("lambda_uniformity", 0.3)
        self.lambda_ortho = EMBEDDING_CONFIG.get("lambda_ortho", 0.05)

    def compute_loss(self, embeddings, features):
        batch_size = embeddings.size(0)
        total_loss = 0.0
        loss_components = {}
        # 1. Distance preservation loss (MDS-style)
        # Preserve pairwise distances from input space to embedding space
        if self.lambda_distance > 0:
            # Input space distances
            input_dists = torch.cdist(features, features, p=2)
            # Embedding space distances
            embed_dists = torch.cdist(embeddings, embeddings, p=2)
            # Normalize by mean input distance to prevent extreme stretching
            scale = input_dists.mean() + 1e-6
            dist_loss = F.mse_loss(embed_dists / scale, input_dists / scale)
            loss_components["distance"] = dist_loss.item()
            total_loss += self.lambda_distance * dist_loss
        # 2. Uniformity loss (repulsion - prevent clustering)
        if self.lambda_uniformity > 0 and batch_size > 1:
            # Pairwise squared distances
            sq_dists = torch.cdist(embeddings, embeddings, p=2).pow(2)
            # Exclude diagonal
            mask = ~torch.eye(batch_size, dtype=torch.bool, device=sq_dists.device)
            # Minimize mean(exp(-2 * dist^2)) to maximize spread
            uniformity_loss = torch.exp(-2.0 * sq_dists[mask]).mean()
            loss_components["uniformity"] = uniformity_loss.item()
            total_loss += self.lambda_uniformity * uniformity_loss
        # 3. Axis decorrelation loss (orthogonality for interpretability)
        if self.lambda_ortho > 0 and batch_size > 1:
            # Compute covariance matrix of embeddings
            emb_centered = embeddings - embeddings.mean(dim=0, keepdim=True)
            cov_matrix = torch.matmul(emb_centered.T, emb_centered) / (batch_size - 1)
            # Penalize off-diagonal elements (want diagonal covariance)
            n_dims = embeddings.size(1)
            ortho_loss = 0.0
            for i in range(n_dims):
                for j in range(i + 1, n_dims):
                    ortho_loss += cov_matrix[i, j].pow(2)
            loss_components["orthogonality"] = ortho_loss.item()
            total_loss += self.lambda_ortho * ortho_loss

        return total_loss, loss_components

    def train_epoch(self, dataloader, optimizer):
        self.model.train()
        total_loss = 0
        all_losses = {"distance": 0, "uniformity": 0, "orthogonality": 0}
        n_batches = 0
        for features in dataloader:
            if isinstance(features, (list, tuple)):
                features = features[0]  # Take anchor from triplet
            features = features.to(self.device)
            optimizer.zero_grad()
            embeddings = self.model(features)
            loss, loss_components = self.compute_loss(embeddings, features)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            for k, v in loss_components.items():
                all_losses[k] += v
            n_batches += 1
        # Average losses
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in all_losses.items()}
        return avg_loss, avg_components

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        all_losses = {"distance": 0, "uniformity": 0, "orthogonality": 0}
        n_batches = 0
        with torch.no_grad():
            for features in dataloader:
                if isinstance(features, (list, tuple)):
                    features = features[0]
                features = features.to(self.device)
                embeddings = self.model(features)
                loss, loss_components = self.compute_loss(embeddings, features)
                total_loss += loss.item()
                for k, v in loss_components.items():
                    all_losses[k] += v
                n_batches += 1
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in all_losses.items()}
        return avg_loss, avg_components


class PickerTrainer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device

    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            enemy_adc, enemy_support, my_support, my_team_cc, enemy_team_cc, target = (
                batch
            )
            enemy_adc = enemy_adc.to(self.device)
            enemy_support = enemy_support.to(self.device)
            my_support = my_support.to(self.device)
            my_team_cc = my_team_cc.to(self.device)
            enemy_team_cc = enemy_team_cc.to(self.device)
            target = target.to(self.device)

            optimizer.zero_grad()
            predicted_emb = self.model(
                enemy_adc, enemy_support, my_support, my_team_cc, enemy_team_cc
            )
            loss = criterion(predicted_emb, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)

    def evaluate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                (
                    enemy_adc,
                    enemy_support,
                    my_support,
                    my_team_cc,
                    enemy_team_cc,
                    target,
                ) = batch
                enemy_adc = enemy_adc.to(self.device)
                enemy_support = enemy_support.to(self.device)
                my_support = my_support.to(self.device)
                my_team_cc = my_team_cc.to(self.device)
                enemy_team_cc = enemy_team_cc.to(self.device)
                target = target.to(self.device)

                predicted_emb = self.model(
                    enemy_adc, enemy_support, my_support, my_team_cc, enemy_team_cc
                )
                loss = criterion(predicted_emb, target)
                total_loss += loss.item()
        return total_loss / len(dataloader)
