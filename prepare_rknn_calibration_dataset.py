from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.signal
import soundfile as sf
import torch
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create RKNN calibration dataset.txt and .npy mel inputs from an audio list.",
    )
    parser.add_argument(
        "--audio-list",
        type=Path,
        required=True,
        help="Text file with one audio path per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("rknn_calibration"),
        help="Output directory for generated .npy files and dataset.txt.",
    )
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    return parser.parse_args()


def _load_audio_paths(list_path: Path) -> list[Path]:
    if not list_path.exists():
        raise FileNotFoundError(f"Audio list not found: {list_path}")

    paths: list[Path] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (list_path.parent / p).resolve()
        paths.append(p)
    return paths


def _audio_to_model_mel(
    audio_path: Path,
    *,
    n_fft: int,
    hop_length: int,
    mel_fb: torch.Tensor,
    window: torch.Tensor,
    target_sr: int,
    target_frames: int,
) -> np.ndarray:
    waveform_np, sr = sf.read(str(audio_path), always_2d=False)
    if waveform_np.ndim == 2:
        waveform_np = waveform_np.mean(axis=1)
    waveform_np = waveform_np.astype(np.float32, copy=False)
    if waveform_np.size == 0:
        raise ValueError("Empty waveform.")

    if sr != target_sr:
        waveform_np = scipy.signal.resample_poly(waveform_np, target_sr, sr).astype(np.float32)

    waveform = torch.from_numpy(waveform_np).unsqueeze(0)  # [1, T]
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        center=True,
        return_complex=True,
    )  # [1, F, T]
    power = stft.abs().pow(2.0)

    # [1, F, T] x [F, M] -> [1, M, T]
    mel = torch.einsum("bft,fm->bmt", power, mel_fb)
    mel = torch.clamp(mel, min=1e-10)
    mel = 10.0 * torch.log10(mel)
    mel = mel - mel.amax(dim=(1, 2), keepdim=True)

    frames = mel.shape[-1]
    if frames > target_frames:
        start = (frames - target_frames) // 2
        mel = mel[:, :, start : start + target_frames]
    elif frames < target_frames:
        pad = target_frames - frames
        pad_value = float(mel.min().item())
        mel = F.pad(mel, (0, pad), value=pad_value)

    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel.to(dtype=torch.float32).numpy()  # [1, n_mels, T]


def main() -> None:
    args = parse_args()
    audio_paths = _load_audio_paths(args.audio_list)
    if not audio_paths:
        raise ValueError(f"No audio paths found in {args.audio_list}")

    npy_dir = args.output_dir / "npy"
    npy_dir.mkdir(parents=True, exist_ok=True)
    dataset_txt = args.output_dir / "dataset.txt"
    target_frames = int(args.duration * args.sr / args.hop_length)
    if target_frames <= 0:
        raise ValueError("Computed target_frames is <= 0. Check duration/sr/hop-length.")

    mel_fb = melscale_fbanks(
        n_freqs=(args.n_fft // 2) + 1,
        f_min=0.0,
        f_max=float(args.sr // 2),
        n_mels=args.n_mels,
        sample_rate=args.sr,
        norm=None,
        mel_scale="htk",
    ).to(dtype=torch.float32)
    window = torch.hann_window(args.n_fft, periodic=True, dtype=torch.float32)

    written: list[str] = []
    failed = 0
    for idx, audio_path in enumerate(audio_paths):
        try:
            mel = _audio_to_model_mel(
                audio_path,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                mel_fb=mel_fb,
                window=window,
                target_sr=args.sr,
                target_frames=target_frames,
            )
        except Exception as exc:
            failed += 1
            print(f"[WARN] Skip {audio_path}: {exc}")
            continue

        out_npy = npy_dir / f"sample_{idx:04d}.npy"
        np.save(out_npy, mel.astype(np.float32))
        written.append(str(out_npy.resolve()))

    if not written:
        raise RuntimeError("No calibration samples generated.")

    dataset_txt.write_text("\n".join(written) + "\n", encoding="utf-8")
    print(f"[OK] Generated samples: {len(written)}")
    print(f"[OK] Failed samples: {failed}")
    print(f"[OK] Calibration dataset list: {dataset_txt.resolve()}")
    print(f"[OK] NPY directory: {npy_dir.resolve()}")


if __name__ == "__main__":
    main()
