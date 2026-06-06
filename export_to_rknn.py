from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

from model.ECAPATDNN import ECAPATDNNBackbone
from utils.data_preprocessing import audio_chunking, audio_to_mel_spectrogram


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export ECAPA-TDNN checkpoint or ONNX model to RKNN. "
            "For INT8 calibration, --dataset may point to dataset.txt, a single "
            "audio file, a directory of audio files, or a text file containing audio paths."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("ECAPATDNN_protonet_model.pth"),
        help="Path to PyTorch checkpoint/state_dict. Ignored when --onnx-model is used.",
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=None,
        help="Use an existing ONNX model instead of exporting one from --checkpoint.",
    )
    parser.add_argument(
        "--onnx-out",
        type=Path,
        default=Path("ecapa_tdnn.onnx"),
        help="Output ONNX model path when exporting from --checkpoint.",
    )
    parser.add_argument(
        "--rknn-out",
        type=Path,
        default=Path("ecapa_tdnn.rknn"),
        help="Output RKNN model path.",
    )
    parser.add_argument(
        "--target-platform",
        type=str,
        default="rv1106",
        help="Rockchip target platform (for example: rv1103, rv1106, rk3566, rk3588).",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Enable INT8 quantization when building RKNN.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Calibration input. Accepts an RKNN dataset.txt of .npy paths, "
            "a single audio file, a directory of audio files, or a text file of audio paths."
        ),
    )
    parser.add_argument(
        "--allow-random-calib",
        action="store_true",
        help="Allow auto-generating random calibration inputs when --dataset is not provided.",
    )
    parser.add_argument(
        "--random-calib-samples",
        type=int,
        default=32,
        help="Number of random samples for auto calibration dataset generation.",
    )
    parser.add_argument(
        "--max-calib-samples",
        type=int,
        default=0,
        help="Maximum number of generated calibration tensors from audio. 0 means no limit.",
    )
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Chunk length in seconds when generating calibration tensors from audio. Default: --duration.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.0,
        help="Chunk overlap in seconds when generating calibration tensors from audio.",
    )
    parser.add_argument("--channels", type=int, default=512)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument(
        "--opset",
        type=int,
        default=12,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--onnx-only",
        action="store_true",
        help="Only export or validate ONNX and skip RKNN conversion.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _maybe_get_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
        for key in ("state_dict", "model_state_dict", "model", "net"):
            state = checkpoint.get(key)
            if isinstance(state, dict) and state and all(torch.is_tensor(v) for v in state.values()):
                return state
    raise ValueError(
        "Unsupported checkpoint format. Expected state_dict or a dict containing "
        "'state_dict'/'model_state_dict'."
    )


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict.keys()):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _compute_input_shape(args: argparse.Namespace) -> tuple[int, int, int]:
    frames = int(args.duration * args.sr / args.hop_length)
    if frames <= 0:
        raise ValueError("Computed frame count is <= 0. Check duration/sr/hop-length.")
    return (1, args.n_mels, frames)


def load_checkpoint_to_model(args: argparse.Namespace) -> ECAPATDNNBackbone:
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = _strip_module_prefix(_maybe_get_state_dict(checkpoint))

    model = ECAPATDNNBackbone(
        n_mels=args.n_mels,
        channels=args.channels,
        emb_dim=args.emb_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_to_onnx(model: ECAPATDNNBackbone, args: argparse.Namespace) -> tuple[Path, tuple[int, int, int]]:
    input_shape = _compute_input_shape(args)
    dummy = torch.randn(*input_shape, dtype=torch.float32)
    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)

    try:
        torch.onnx.export(
            model,
            dummy,
            str(args.onnx_out),
            input_names=["mel"],
            output_names=["embedding"],
            export_params=True,
            do_constant_folding=True,
            opset_version=args.opset,
        )
    except ModuleNotFoundError as exc:
        if exc.name in {"onnx", "onnxscript"}:
            raise ModuleNotFoundError(
                f"Missing dependency '{exc.name}'. Install ONNX dependencies first, "
                "for example: pip install onnx onnxscript"
            ) from exc
        raise

    print(f"[OK] ONNX exported: {args.onnx_out}")
    print(f"     input shape: {input_shape}")
    return args.onnx_out, input_shape


def _create_random_calibration_dataset(
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
) -> Path:
    if args.random_calib_samples <= 0:
        raise ValueError("--random-calib-samples must be > 0.")

    calib_dir = args.rknn_out.parent / "rknn_random_calib"
    npy_dir = calib_dir / "npy"
    dataset_txt = calib_dir / "dataset.txt"
    npy_dir.mkdir(parents=True, exist_ok=True)

    lines = []
    for i in range(args.random_calib_samples):
        sample = np.random.normal(loc=0.0, scale=1.0, size=input_shape).astype(np.float32)
        npy_path = npy_dir / f"sample_{i:04d}.npy"
        np.save(npy_path, sample)
        lines.append(str(npy_path.resolve()))

    dataset_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[WARN] Using random calibration dataset: {dataset_txt}")
    print("[WARN] For better accuracy, pass --dataset with real audio from your use case.")
    return dataset_txt


def _patch_onnx_mapping_for_rknn() -> None:
    """
    RKNN toolkit2 still expects `onnx.mapping`, which was removed in newer ONNX.
    Build a small compatibility shim from `onnx._mapping`.
    """
    try:
        import onnx
    except ImportError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'onnx'. Install ONNX first, for example: "
            "pip install onnx onnxscript"
        ) from exc

    mapping = getattr(onnx, "mapping", None)
    if mapping is not None and hasattr(mapping, "NP_TYPE_TO_TENSOR_TYPE"):
        return

    tensor_type_map = getattr(getattr(onnx, "_mapping", None), "TENSOR_TYPE_MAP", None)
    if tensor_type_map is None:
        raise RuntimeError(
            "Cannot create ONNX compatibility mapping required by RKNN. "
            "Please use ONNX 1.x with `_mapping.TENSOR_TYPE_MAP` available."
        )

    tensor_to_np: dict[int, np.dtype] = {}
    np_to_tensor: dict[object, int] = {}
    for tensor_type, info in tensor_type_map.items():
        np_dtype = np.dtype(info.np_dtype)
        tensor_to_np[tensor_type] = np_dtype
        np_to_tensor[np_dtype] = tensor_type
        np_to_tensor[np_dtype.type] = tensor_type

    onnx.mapping = SimpleNamespace(  # type: ignore[attr-defined]
        TENSOR_TYPE_TO_NP_TYPE=tensor_to_np,
        NP_TYPE_TO_TENSOR_TYPE=np_to_tensor,
    )
    print("[INFO] Applied ONNX compatibility shim for RKNN (onnx.mapping).")


def _is_audio_path(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_SUFFIXES


def _read_list_file(list_path: Path) -> list[str]:
    lines = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        lines.append(item)
    return lines


def _resolve_list_entry(list_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = (list_path.parent / path).resolve()
    return path


def _is_prebuilt_npy_dataset(dataset_path: Path) -> bool:
    if not dataset_path.is_file() or dataset_path.suffix.lower() != ".txt":
        return False

    lines = _read_list_file(dataset_path)
    if not lines:
        return False

    for raw_path in lines:
        entry = _resolve_list_entry(dataset_path, raw_path)
        if entry.suffix.lower() != ".npy" or not entry.exists():
            return False
    return True


def _normalize_npy_dataset(dataset_path: Path, output_dir: Path) -> Path:
    lines = _read_list_file(dataset_path)
    normalized_entries = []
    for raw_path in lines:
        entry = _resolve_list_entry(dataset_path, raw_path)
        if not entry.exists():
            raise FileNotFoundError(f"Calibration tensor not found: {entry}")
        normalized_entries.append(str(entry.resolve()))

    normalized_path = output_dir / f"{dataset_path.stem}_absolute.txt"
    normalized_path.write_text("\n".join(normalized_entries) + "\n", encoding="utf-8")
    return normalized_path


def _collect_audio_paths(dataset_source: Path) -> list[Path]:
    if not dataset_source.exists():
        raise FileNotFoundError(f"Dataset source not found: {dataset_source}")

    if dataset_source.is_dir():
        audio_paths = sorted(
            path.resolve()
            for path in dataset_source.rglob("*")
            if path.is_file() and _is_audio_path(path)
        )
        if not audio_paths:
            raise ValueError(f"No audio files found in directory: {dataset_source}")
        return audio_paths

    if _is_audio_path(dataset_source):
        return [dataset_source.resolve()]

    if dataset_source.suffix.lower() == ".txt":
        lines = _read_list_file(dataset_source)
        if not lines:
            raise ValueError(f"No entries found in list file: {dataset_source}")

        audio_paths = []
        for raw_path in lines:
            path = _resolve_list_entry(dataset_source, raw_path)
            if not path.exists():
                raise FileNotFoundError(f"Audio file listed in {dataset_source} was not found: {path}")
            if not _is_audio_path(path):
                raise ValueError(
                    f"List file {dataset_source} contains non-audio entry: {raw_path}. "
                    "Use a dataset.txt of .npy files or a list of audio files."
                )
            audio_paths.append(path)
        return audio_paths

    raise ValueError(
        f"Unsupported dataset source: {dataset_source}. "
        "Expected dataset.txt, an audio list .txt, a directory, or an audio file."
    )


def _build_audio_calibration_dataset(
    args: argparse.Namespace,
    dataset_source: Path,
    input_shape: tuple[int, int, int],
) -> Path:
    audio_paths = _collect_audio_paths(dataset_source)
    calib_dir = args.rknn_out.parent / f"{args.rknn_out.stem}_audio_calib"
    npy_dir = calib_dir / "npy"
    dataset_txt = calib_dir / "dataset.txt"
    npy_dir.mkdir(parents=True, exist_ok=True)

    chunk_duration = args.chunk_duration if args.chunk_duration is not None else args.duration
    if chunk_duration <= 0:
        raise ValueError("--chunk-duration must be > 0.")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be >= 0.")

    written: list[str] = []
    failed = 0

    for audio_idx, audio_path in enumerate(audio_paths):
        try:
            _, waveform, sr_loaded = audio_to_mel_spectrogram(
                audio_path=audio_path,
                sr=args.sr,
                duration=None,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                n_mels=args.n_mels,
                to_db=True,
            )
            mel_chunks = audio_chunking(
                waveform,
                sr_loaded,
                chunk_duration=chunk_duration,
                overlap_duration=args.chunk_overlap,
                return_mels=True,
                n_mels=args.n_mels,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                target_duration=args.duration,
            )
        except Exception as exc:
            failed += 1
            print(f"[WARN] Skip calibration audio {audio_path}: {exc}")
            continue

        if not mel_chunks:
            failed += 1
            print(f"[WARN] Skip calibration audio {audio_path}: no usable chunks.")
            continue

        for chunk_idx, mel_chunk in enumerate(mel_chunks):
            mel_np = mel_chunk.cpu().numpy().astype(np.float32, copy=False)
            if mel_np.shape != input_shape:
                raise RuntimeError(
                    f"Calibration tensor shape mismatch for {audio_path}: "
                    f"expected {input_shape}, got {mel_np.shape}"
                )

            out_npy = npy_dir / f"sample_{audio_idx:04d}_{chunk_idx:02d}.npy"
            np.save(out_npy, mel_np)
            written.append(str(out_npy.resolve()))

            if args.max_calib_samples > 0 and len(written) >= args.max_calib_samples:
                break

        if args.max_calib_samples > 0 and len(written) >= args.max_calib_samples:
            break

    if not written:
        raise RuntimeError("No calibration tensors were generated from the provided audio input.")

    dataset_txt.write_text("\n".join(written) + "\n", encoding="utf-8")
    print(f"[OK] Generated audio calibration dataset: {dataset_txt}")
    print(f"     audio files scanned: {len(audio_paths)}")
    print(f"     tensors written: {len(written)}")
    print(f"     failed audio files: {failed}")
    return dataset_txt


def _resolve_calibration_dataset(
    args: argparse.Namespace,
    input_shape: tuple[int, int, int],
) -> Path | None:
    if not args.quantize:
        return None

    if args.dataset is None:
        if args.allow_random_calib:
            return _create_random_calibration_dataset(args, input_shape)
        raise ValueError(
            "--dataset is required when --quantize is enabled. "
            "It may be dataset.txt, a single .wav, a directory of audio, "
            "or a .txt list of audio files. If you only want a quick test build, "
            "add --allow-random-calib."
        )

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset source not found: {args.dataset}")

    if _is_prebuilt_npy_dataset(args.dataset):
        args.rknn_out.parent.mkdir(parents=True, exist_ok=True)
        return _normalize_npy_dataset(args.dataset, args.rknn_out.parent)

    return _build_audio_calibration_dataset(args, args.dataset, input_shape)


def export_to_rknn(
    args: argparse.Namespace,
    onnx_model_path: Path,
    input_shape: tuple[int, int, int],
) -> None:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise ImportError(
            "rknn-toolkit2 is not installed in this environment. "
            "Install it first, then rerun the script."
        ) from exc

    target = args.target_platform.lower()
    rv110x_targets = {"rv1103", "rv1106"}
    if target in rv110x_targets and not args.quantize:
        if args.allow_random_calib:
            print(f"[WARN] {args.target_platform} requires INT8 quantization. Auto-enabling --quantize.")
            args.quantize = True
        else:
            raise ValueError(
                f"{args.target_platform} requires INT8 quantization in rknn-toolkit2. "
                "Rerun with --quantize --dataset <audio_or_dataset.txt> "
                "or use --allow-random-calib."
            )

    dataset_path = _resolve_calibration_dataset(args, input_shape)
    _patch_onnx_mapping_for_rknn()

    rknn = RKNN(verbose=args.verbose)
    ret = rknn.config(target_platform=args.target_platform)
    if ret != 0:
        raise RuntimeError(f"rknn.config failed with code: {ret}")

    try:
        ret = rknn.load_onnx(model=str(onnx_model_path))
    except Exception:
        ret = rknn.load_onnx(
            model=str(onnx_model_path),
            inputs=["mel"],
            input_size_list=[list(input_shape)],
            outputs=["embedding"],
        )
    if ret != 0:
        raise RuntimeError(
            f"rknn.load_onnx failed with code: {ret}. "
            "If this fails on Conv1d ops, consider converting Conv1d blocks to Conv2d-first."
        )

    ret = rknn.build(
        do_quantization=args.quantize,
        dataset=str(dataset_path) if dataset_path is not None else None,
    )
    if ret != 0:
        raise RuntimeError(f"rknn.build failed with code: {ret}")

    args.rknn_out.parent.mkdir(parents=True, exist_ok=True)
    ret = rknn.export_rknn(str(args.rknn_out))
    if ret != 0:
        raise RuntimeError(f"rknn.export_rknn failed with code: {ret}")
    rknn.release()

    print(f"[OK] RKNN exported: {args.rknn_out}")
    print(f"     target platform: {args.target_platform}")
    print(f"     quantized: {args.quantize}")
    if dataset_path is not None:
        print(f"     calibration dataset: {dataset_path}")


def main() -> None:
    args = parse_args()
    input_shape = _compute_input_shape(args)

    if args.onnx_model is not None:
        if not args.onnx_model.exists():
            raise FileNotFoundError(f"ONNX model not found: {args.onnx_model}")
        onnx_model_path = args.onnx_model
        print(f"[INFO] Using existing ONNX model: {onnx_model_path}")
        print(f"       expected input shape: {input_shape}")
    else:
        model = load_checkpoint_to_model(args)
        onnx_model_path, input_shape = export_to_onnx(model, args)

    if args.onnx_only:
        print("[DONE] ONNX step complete (RKNN step skipped by --onnx-only).")
        return

    export_to_rknn(args, onnx_model_path, input_shape)
    print("[DONE] ONNX -> RKNN conversion complete.")


if __name__ == "__main__":
    main()
