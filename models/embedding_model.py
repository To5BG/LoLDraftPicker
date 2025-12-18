import torch.nn as nn


class ChampionEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dims, embedding_dim, dropout=0.2):
        super(ChampionEmbedding, self).__init__()
        layers = []
        prev_dim = input_dim
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        # Final embedding layers
        layers.append(nn.Linear(prev_dim, embedding_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
