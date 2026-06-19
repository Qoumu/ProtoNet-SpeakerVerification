from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model.XVector import XVectorBackbone
from utils.paths import get_project_root


PROJECT_ROOT = get_project_root()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize an x-vector backbone checkpoint for later training.",
    )
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output" / "xvector_init.pth")
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--tdnn-channels", type=int, default=512)
    parser.add_argument("--stats-channels", type=int, default=1500)
    parser.add_argument("--embedding-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model = XVectorBackbone(
        n_mels=args.n_mels,
        tdnn_channels=args.tdnn_channels,
        stats_channels=args.stats_channels,
        emb_dim=args.embedding_dim,
        dropout=args.dropout,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    total_params = sum(param.numel() for param in model.parameters())
    print(f"[OK] Saved x-vector initial checkpoint: {args.output.resolve()}")
    print(f"[OK] Parameters: {total_params:,}")
    print(f"[OK] Embedding dim: {args.embedding_dim}")


if __name__ == "__main__":
    main()
