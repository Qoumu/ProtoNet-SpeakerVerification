import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Small building blocks
# ----------------------------
class TDNNBlock(nn.Module):
    """
    1D conv over time. Input: (B, C, T) -> (B, out_ch, T)
    """
    def __init__(self, in_ch, out_ch, kernel_size=5, dilation=1, groups=1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                              padding=padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """
    Squeeze-Excitation over channel dim. Input: (B, C, T)
    """
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        hidden = max(8, int(channels * se_ratio))
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x):
        s = x.mean(dim=2, keepdim=True)          # (B, C, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class Res2Block(nn.Module):
    """
    Res2Net-style split conv for ECAPA.
    Input/Output: (B, C, T)
    """
    def __init__(self, channels, scale=8, kernel_size=3, dilation=2):
        super().__init__()
        assert channels % scale == 0, "channels must be divisible by scale"
        self.scale = scale
        self.width = channels // scale

        self.convs = nn.ModuleList([
            nn.Conv1d(self.width, self.width, kernel_size,
                      padding=((kernel_size - 1) // 2) * dilation,
                      dilation=dilation, bias=False)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(scale - 1)])
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # split channels into scale chunks
        xs = torch.split(x, self.width, dim=1)
        y = [xs[0]]
        s = xs[1]
        for i in range(self.scale - 1):
            if i > 0:
                s = s + xs[i + 1]
            s = self.act(self.bns[i](self.convs[i](s)))
            y.append(s)
        return torch.cat(y, dim=1)


class ECAPARes2SE(nn.Module):
    """
    A typical ECAPA block: 1x1 -> Res2 -> 1x1 -> SE -> residual
    """
    def __init__(self, channels, scale=8, kernel_size=3, dilation=2):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.res2 = Res2Block(channels, scale=scale, kernel_size=kernel_size, dilation=dilation)
        self.post = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(channels)

    def forward(self, x):
        r = x
        x = self.pre(x)
        x = self.res2(x)
        x = self.post(x)
        x = self.se(x)
        return x + r


class AttentiveStatsPooling(nn.Module):
    """
    Attentive statistics pooling (mean+std) with attention over time.
    Input: (B, C, T) -> Output: (B, 2C)
    """
    def __init__(self, channels, attn_channels=128):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(channels, attn_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(attn_channels),
            nn.Conv1d(attn_channels, channels, kernel_size=1),
        )

    def forward(self, x):
        # attention weights over time
        e = self.tdnn(x)                 # (B, C, T)
        a = torch.softmax(e, dim=2)      # (B, C, T)

        # weighted mean
        mu = torch.sum(a * x, dim=2)     # (B, C)

        # weighted std (stable)
        x2 = torch.sum(a * (x ** 2), dim=2)
        var = torch.clamp(x2 - mu ** 2, min=1e-9)
        std = torch.sqrt(var)            # (B, C)

        return torch.cat([mu, std], dim=1)  # (B, 2C)


# ----------------------------
# ECAPA-TDNN backbone
# ----------------------------
class ECAPATDNNBackbone(nn.Module):
    """
    Input mel: (B, F, T) where F = n_mels
    Output embedding: (B, emb_dim) L2-normalized
    """
    def __init__(
        self,
        n_mels=80,
        channels=512,
        emb_dim=192,
        scale=8
    ):
        super().__init__()

        # Frame-level
        self.layer1 = TDNNBlock(n_mels, channels, kernel_size=5, dilation=1)
        self.layer2 = ECAPARes2SE(channels, scale=scale, kernel_size=3, dilation=2)
        self.layer3 = ECAPARes2SE(channels, scale=scale, kernel_size=3, dilation=3)
        self.layer4 = ECAPARes2SE(channels, scale=scale, kernel_size=3, dilation=4)
        # self.layer5 = ECAPARes2SE(channels, scale=scale, kernel_size=3, dilation=5)

        # Multi-layer feature aggregation (common in ECAPA)
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

        # Pooling
        self.pool = AttentiveStatsPooling(channels, attn_channels=128)

        # Utterance-level projection
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, emb_dim, bias=False),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, mel):
        """
        mel: (B, F, T)
        """
        assert mel.dim() == 3, "Expected mel with shape (B, F, T)"
        x = mel  # (B, F, T)

        x1 = self.layer1(x)      # (B, C, T)
        x2 = self.layer2(x1)     # (B, C, T)
        x3 = self.layer3(x2)     # (B, C, T)
        x4 = self.layer4(x3)     # (B, C, T)
        # x5 = self.layer5(x4)     # (B, C, T)

        # concatenate last 3 blocks (typical ECAPA)
        x_cat = torch.cat([x2, x3, x4], dim=1)  # (B, 3C, T)
        x_mfa = self.mfa(x_cat)                 # (B, C, T)

        stats = self.pool(x_mfa)                # (B, 2C)
        emb = self.fc(stats)                    # (B, emb_dim)

        # L2 normalize for metric learning / ProtoNet
        emb = F.normalize(emb, p=2, dim=1)
        return emb
    
    def pn_predict(self, query_embeddings, prototypes):
        """Return cosine-similarity scores against each class prototype."""
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        return torch.matmul(query_embeddings, prototypes.T)
