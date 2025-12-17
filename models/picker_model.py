import torch
import torch.nn as nn


class DraftPicker(nn.Module):
    def __init__(self, embedding_dim, hidden_dims, output_dim, dropout=0.3):
        super(DraftPicker, self).__init__()
        self.embedding_dim = embedding_dim
        # Input: 3 champion embeddings + 4 aggregate features
        input_dim = 3 * embedding_dim + 4
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        # Final output layers
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        enemy_adc_emb,
        enemy_support_emb,
        my_support_emb,
        my_team_cc,
        enemy_team_cc,
    ):
        context = torch.cat(
            [
                enemy_adc_emb,
                enemy_support_emb,
                my_support_emb,
                my_team_cc,
                enemy_team_cc,
            ],
            dim=1,
        )
        return self.network(context)
