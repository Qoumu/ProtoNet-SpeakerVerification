from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class TDNNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(values)))


class SEBlock(nn.Module):
    def __init__(self, channels: int, ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(channels * ratio))
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        scale = values.mean(dim=2, keepdim=True)
        scale = functional.relu(self.fc1(scale), inplace=True)
        return values * torch.sigmoid(self.fc2(scale))


class Res2Block(nn.Module):
    def __init__(self, channels: int, scale: int = 8, dilation: int = 2):
        super().__init__()
        if channels % scale:
            raise ValueError("channels must be divisible by scale")
        self.scale = scale
        self.width = channels // scale
        self.convs = nn.ModuleList(
            nn.Conv1d(
                self.width,
                self.width,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            )
            for _ in range(scale - 1)
        )
        self.bns = nn.ModuleList(nn.BatchNorm1d(self.width) for _ in range(scale - 1))
        self.act = nn.ReLU(inplace=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(values, self.width, dim=1)
        output = [chunks[0]]
        current = chunks[1]
        for index in range(self.scale - 1):
            if index:
                current = current + chunks[index + 1]
            current = self.act(self.bns[index](self.convs[index](current)))
            output.append(current)
        return torch.cat(output, dim=1)


class ECAPARes2SE(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.res2 = Res2Block(channels, dilation=dilation)
        self.post = nn.Sequential(
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.se(self.post(self.res2(self.pre(values)))) + values


class AttentiveStatsPooling(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.tdnn = nn.Sequential(
            nn.Conv1d(channels, 128, 1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, channels, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        attention = torch.softmax(self.tdnn(values), dim=2)
        mean = torch.sum(attention * values, dim=2)
        second_moment = torch.sum(attention * values**2, dim=2)
        std = torch.sqrt(torch.clamp(second_moment - mean**2, min=1e-9))
        return torch.cat((mean, std), dim=1)


class ECAPATDNNBackbone(nn.Module):
    """Architecture matching the repository's trained ECAPA-TDNN checkpoint."""

    def __init__(self, n_mels: int = 80, channels: int = 512, embedding_dimension: int = 192):
        super().__init__()
        self.layer1 = TDNNBlock(n_mels, channels)
        self.layer2 = ECAPARes2SE(channels, dilation=2)
        self.layer3 = ECAPARes2SE(channels, dilation=3)
        self.layer4 = ECAPARes2SE(channels, dilation=4)
        self.mfa = nn.Sequential(
            nn.Conv1d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.pool = AttentiveStatsPooling(channels)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, embedding_dimension, bias=False),
            nn.BatchNorm1d(embedding_dimension),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        first = self.layer1(mel)
        second = self.layer2(first)
        third = self.layer3(second)
        fourth = self.layer4(third)
        merged = self.mfa(torch.cat((second, third, fourth), dim=1))
        return functional.normalize(self.fc(self.pool(merged)), p=2, dim=1)
