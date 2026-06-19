import torch
import torch.nn as nn
import torch.nn.functional as F


class StatisticsPooling(nn.Module):
    """Mean and standard-deviation pooling over the time axis."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=2)
        std = x.std(dim=2, unbiased=False).clamp_min(1e-6)
        return torch.cat([mean, std], dim=1)


class XVectorBackbone(nn.Module):
    """
    X-vector TDNN speaker embedding backbone.

    Input mel shape: (B, n_mels, T)
    Output embedding shape: (B, emb_dim), L2-normalized
    """

    def __init__(
        self,
        n_mels: int = 80,
        tdnn_channels: int = 512,
        stats_channels: int = 1500,
        emb_dim: int = 192,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.frame_layers = nn.Sequential(
            nn.Conv1d(n_mels, tdnn_channels, kernel_size=5, dilation=1, padding=2),
            nn.BatchNorm1d(tdnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(tdnn_channels, tdnn_channels, kernel_size=3, dilation=2, padding=2),
            nn.BatchNorm1d(tdnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(tdnn_channels, tdnn_channels, kernel_size=3, dilation=3, padding=3),
            nn.BatchNorm1d(tdnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(tdnn_channels, tdnn_channels, kernel_size=1),
            nn.BatchNorm1d(tdnn_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(tdnn_channels, stats_channels, kernel_size=1),
            nn.BatchNorm1d(stats_channels),
            nn.ReLU(inplace=True),
        )
        self.pool = StatisticsPooling()
        self.segment = nn.Sequential(
            nn.Linear(stats_channels * 2, tdnn_channels),
            nn.BatchNorm1d(tdnn_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(tdnn_channels, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        if mel.dim() != 3:
            raise ValueError("Expected mel with shape (B, n_mels, T)")

        x = self.frame_layers(mel)
        x = self.pool(x)
        embedding = self.segment(x)
        return F.normalize(embedding, p=2, dim=1)

    def pn_predict(self, query_embeddings: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return torch.matmul(query_embeddings, prototypes.T)
