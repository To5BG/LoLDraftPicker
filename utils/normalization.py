import torch
import numpy as np


class EmbeddingNormalizer:
    def __init__(
        self, method="percentile", percentile_range=(2, 98), target_range=(-1, 1)
    ):
        self.method = method
        self.percentile_range = percentile_range
        self.target_range = target_range
        # Will be computed from data
        self.center = None  # Global center point
        self.radius = None  # Single global radius (isotropic scaling)
        self.is_fitted = False

    def fit(self, embeddings):
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        # Global center (mean)
        self.center = embeddings.mean(axis=0)
        # Radial distances from center
        centered = embeddings - self.center
        radii = np.linalg.norm(centered, axis=1)
        # Robust radius using percentile (not per-axis!)
        if self.method == "percentile":
            self.radius = np.percentile(radii, self.percentile_range[1])
        else:  # max
            self.radius = radii.max()

        if self.radius < 1e-8:
            self.radius = 1.0

        self.is_fitted = True

    def transform(self, embeddings, clip=True):
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        is_torch = isinstance(embeddings, torch.Tensor)
        if is_torch:
            device = embeddings.device
            embeddings = embeddings.cpu().numpy()
        # Center and scale isotropically
        z_norm = (embeddings - self.center) / self.radius
        if clip:
            # Radial clipping (preserves angles, bounds to unit ball)
            norms = np.linalg.norm(z_norm, axis=-1, keepdims=True)
            mask = norms.squeeze() > 1.0
            if mask.any():
                z_norm[mask] = z_norm[mask] / norms[mask]
        # Map to target range
        target_min, target_max = self.target_range
        if target_min != -1 or target_max != 1:
            # Linear map from [-1,1] to [target_min, target_max]
            z_norm = (z_norm + 1) / 2  # to [0, 1]
            z_norm = z_norm * (target_max - target_min) + target_min
        if is_torch:
            z_norm = torch.from_numpy(z_norm).float().to(device)
        return z_norm

    def inverse_transform(self, normalized_embeddings):
        if not self.is_fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")
        is_torch = isinstance(normalized_embeddings, torch.Tensor)
        if is_torch:
            device = normalized_embeddings.device
            normalized_embeddings = normalized_embeddings.cpu().numpy()
        z = normalized_embeddings.copy()
        # Map from target range to [-1, 1]
        target_min, target_max = self.target_range
        if target_min != -1 or target_max != 1:
            z = (z - target_min) / (target_max - target_min)  # to [0, 1]
            z = z * 2 - 1  # to [-1, 1]
        # Unscale and uncenter (isotropically)
        z = z * self.radius + self.center
        if is_torch:
            z = torch.from_numpy(z).float().to(device)
        return z

    def get_params(self):
        return {
            "method": self.method,
            "percentile_range": self.percentile_range,
            "target_range": self.target_range,
            "center": self.center.tolist() if self.center is not None else None,
            "radius": float(self.radius) if self.radius is not None else None,
            "is_fitted": self.is_fitted,
        }

    def set_params(self, params):
        self.method = params["method"]
        self.percentile_range = tuple(params["percentile_range"])
        self.target_range = tuple(params["target_range"])
        self.center = (
            np.array(params["center"]) if params["center"] is not None else None
        )
        self.radius = params["radius"]
        self.is_fitted = params["is_fitted"]


def load_normalizer(filepath):
    params = torch.load(filepath)
    normalizer = EmbeddingNormalizer()
    normalizer.set_params(params)
    return normalizer
