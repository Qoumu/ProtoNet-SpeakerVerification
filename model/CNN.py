import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import *

class SpecAugment(nn.Module):
    """
    SpecAugment: Data augmentation for spectrograms
    Masks time and frequency bands to improve generalization
    """
    def __init__(self, freq_mask_param=15, time_mask_param=35, n_freq_masks=2, n_time_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, mel_spec):
        """
        Args:
            mel_spec: [batch, n_mels, time]
        """
        if not self.training:
            return mel_spec

        mel_spec = mel_spec.clone()

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = torch.randint(0, self.freq_mask_param, (1,))
            f0 = torch.randint(0, mel_spec.size(1) - f, (1,))
            mel_spec[:, f0:f0+f, :] = mel_spec.min()  # Use min instead of 0 for dB scale

        # Time masking
        for _ in range(self.n_time_masks):
            t = torch.randint(0, min(self.time_mask_param, mel_spec.size(2)), (1,))
            if mel_spec.size(2) - t > 0:
                t0 = torch.randint(0, mel_spec.size(2) - t, (1,))
                mel_spec[:, :, t0:t0+t] = mel_spec.min()

        return mel_spec

class SpeakerCNN(nn.Module):
    """
    Lightweight CNN for speaker classification with limited data

    Key design choices for limited data:
    - Smaller model (fewer parameters to avoid overfitting)
    - Batch normalization for stability
    - Dropout for regularization
    - Global average pooling to reduce parameters
    - Residual connections for better gradient flow
    """

    def __init__(self, n_mels=80, dropout=0.2):
        super().__init__()

        self.spec_augment = SpecAugment()

        # First block - extract low-level features
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout * 0.5)
        )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2, 2),
        #     nn.Dropout2d(dropout * 0.5)
        # )

        # Seccond conv block - high-level features
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout)
        )

        # Fourth block - speaker-specific patterns

        # self.conv7 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        # )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Embedding layer - creates speaker representation
        self.embedding = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_embedding=False):
        """
        Args:
            x: mel spectrogram [batch, n_mels, time]
            return_embedding: if True, returns embedding instead of logits

        Returns:
            logits [batch, n_speakers] or embeddings [batch, 128]
        """
        # Add channel dimension: [batch, 1, n_mels, time]
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # # Apply SpecAugment during training
        # if self.training:
        #     x = self.spec_augment(x.squeeze(1)).unsqueeze(1)

        # Convolutional feature extraction
        x = self.conv1(x)  # [batch, 64, n_mels/2, time/2]
        x = self.conv2(x)  # [batch, 128, n_mels/4, time/4]
        x = F.relu(x + self.conv3(x))  # [batch, 128, n_mels/8, time/8]
        # x = self.conv4(x)  # [batch, 256, n_mels/16, time/16]
        x = self.conv5(x)  # [batch, 512, 1, 1]
        x = F.relu(x + self.conv6(x))
        # x = self.conv7(x)
        x = self.global_pool(x)  # [batch, 512, 1, 1]

        # Flatten
        x = x.view(x.size(0), -1)  # [batch, 512]

        # Get embedding
        embedding = self.embedding(x)  # [batch, 64]

        return embedding

    def pn_predict(self, query_embeddings, prototypes):
        # Compute distances to all prototypes
        distances = cosine_distance(query_embeddings, prototypes)

        # Convert distances to logits (negative distances)
        logits = -distances

        return logits
