from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare an input .wav file against a .wav or .pt reference with the exported ECAPA-TDNN ONNX model.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "output" / "ecapa_tdnn.onnx",
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--input-audio",
        type=Path,
        required=True,
        help="Input/query .wav file.",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        required=True,
        help="Reference `.wav` file or `.pt` embedding store.",
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Audio sample rate used for preprocessing.",
    )
    parser.add_argument(
        "--n-mels",
        type=int,
        default=80,
        help="Mel bin count used for preprocessing.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Target audio duration in seconds after pad/crop.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=1024,
        help="FFT size used for mel extraction.",
    )
    parser.add_argument(
        "--hop-length",
        type=int,
        default=256,
        help="Hop size used for mel extraction.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="ONNX Runtime execution providers.",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.7,
        help="Minimum cosine similarity required to accept the best match.",
    )
    return parser.parse_args()


def _pad_or_crop_mel(mel: np.ndarray, *, target_frames: int) -> np.ndarray:
    current_frames = mel.shape[1]
    if current_frames > target_frames:
        start = (current_frames - target_frames) // 2
        mel = mel[:, start : start + target_frames]
    elif current_frames < target_frames:
        mel = np.pad(
            mel,
            ((0, 0), (0, target_frames - current_frames)),
            mode="constant",
            constant_values=float(mel.min()),
        )
    return mel


def _preprocess_audio(
    audio_path: Path,
    *,
    sr: int,
    n_mels: int,
    duration: float,
    n_fft: int,
    hop_length: int,
) -> np.ndarray:
    from utils.data_preprocessing import audio_to_mel_spectrogram

    mel, _, sr_loaded = audio_to_mel_spectrogram(
        audio_path=audio_path,
        sr=sr,
        duration=None,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        to_db=True,
    )

    target_frames = int(duration * sr_loaded / hop_length)
    mel = _pad_or_crop_mel(mel, target_frames=target_frames)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    return mel.astype(np.float32)[None, :, :]


def _coerce_input_shape(array: np.ndarray, expected_shape: Sequence[int | str | None]) -> np.ndarray:
    if array.ndim == 2:
        array = array[None, :, :]
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 2D or 3D input tensor, got shape {array.shape}.",
        )

    normalized = array.astype(np.float32, copy=False)

    if len(expected_shape) != normalized.ndim:
        raise ValueError(
            f"Model expects {len(expected_shape)} dims, but input has {normalized.ndim}: {normalized.shape}.",
        )

    for idx, (actual, expected) in enumerate(zip(normalized.shape, expected_shape)):
        if isinstance(expected, int) and expected > 0 and actual != expected:
            raise ValueError(
                f"Input shape mismatch at axis {idx}: expected {expected_shape}, got {normalized.shape}.",
            )

    return normalized


def _cosine_similarity(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x_flat = np.asarray(x, dtype=np.float32).reshape(-1)
    y_flat = np.asarray(y, dtype=np.float32).reshape(-1)

    x_norm = np.linalg.norm(x_flat)
    y_norm = np.linalg.norm(y_flat)
    denom = max(x_norm * y_norm, eps)
    return float(np.dot(x_flat, y_flat) / denom)


def _log_softmax(values: np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    if logits.ndim != 1:
        raise ValueError(f"Expected a 1D score vector, got shape {logits.shape}.")

    max_logit = np.max(logits)
    shifted = logits - max_logit
    logsumexp = max_logit + np.log(np.exp(shifted).sum())
    return logits - logsumexp


def _coerce_embedding_shape(array: np.ndarray) -> np.ndarray:
    normalized = np.asarray(array, dtype=np.float32)
    if normalized.ndim == 1:
        return normalized.reshape(1, -1)
    if normalized.ndim == 2 and normalized.shape[0] == 1:
        return normalized
    raise ValueError(
        f"Expected a 1D embedding or a single-row 2D embedding, got shape {normalized.shape}.",
    )


def _embedding_array_from_payload(payload: Any, *, reference_path: Path) -> np.ndarray:
    if hasattr(payload, "detach"):
        return np.asarray(payload.detach().cpu(), dtype=np.float32)
    if isinstance(payload, np.ndarray):
        return payload.astype(np.float32, copy=False)
    if isinstance(payload, (list, tuple)):
        return np.asarray(payload, dtype=np.float32)
    if isinstance(payload, dict) and "embedding" in payload:
        return _embedding_array_from_payload(payload["embedding"], reference_path=reference_path)
    raise ValueError(
        f"Unsupported .pt reference payload type {type(payload).__name__} in {reference_path}."
    )


def _extract_embeddings_from_pt(
    payload: Any,
    *,
    reference_path: Path,
) -> list[tuple[str, np.ndarray]]:
    if isinstance(payload, dict):
        entries: list[tuple[str, np.ndarray]] = []
        for key, value in payload.items():
            try:
                embedding = _coerce_embedding_shape(
                    _embedding_array_from_payload(value, reference_path=reference_path)
                )
            except ValueError:
                continue
            entries.append((str(key), embedding))

        if entries:
            embedding_dims = sorted({embedding.shape[1] for _, embedding in entries})
            if len(entries) > 1 and len(embedding_dims) != 1:
                dims_preview = ", ".join(map(str, embedding_dims[:10]))
                raise ValueError(
                    f"{reference_path} does not look like an enrolled embedding store. "
                    f"Found inconsistent candidate embedding dimensions: {dims_preview}"
                )
            return entries

        if "embedding" in payload:
            return [
                (
                    f"{reference_path.name}:embedding",
                    _coerce_embedding_shape(
                        _embedding_array_from_payload(payload["embedding"], reference_path=reference_path)
                    ),
                )
            ]

        available = ", ".join(map(str, list(payload.keys())[:10]))
        raise ValueError(
            f"No usable embeddings found in {reference_path}. Available entries: {available}"
        )

    return [
        (
            reference_path.name,
            _coerce_embedding_shape(_embedding_array_from_payload(payload, reference_path=reference_path)),
        )
    ]


def _load_reference_embeddings(
    reference_path: Path,
    *,
    session: "ort.InferenceSession",
    input_name: str,
    expected_shape: Sequence[int | str | None],
    args: argparse.Namespace,
) -> list[tuple[str, np.ndarray]]:
    suffix = reference_path.suffix.lower()
    if suffix == ".wav":
        reference_tensor = _coerce_input_shape(
            _preprocess_audio(
                reference_path,
                sr=args.sr,
                n_mels=args.n_mels,
                duration=args.duration,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
            ),
            expected_shape,
        )
        print(f"[INFO] reference audio: {reference_path}")
        print(
            "[INFO] reference tensor stats: "
            f"shape={reference_tensor.shape} dtype={reference_tensor.dtype} "
            f"min={reference_tensor.min():.4f} max={reference_tensor.max():.4f}"
        )
        reference_embedding = np.asarray(
            session.run(None, {input_name: reference_tensor})[0],
            dtype=np.float32,
        )
        return [(str(reference_path), reference_embedding)]

    if suffix == ".pt":
        try:
            import torch
        except ImportError as exc:
            raise SystemExit(
                "Loading a .pt reference requires PyTorch. Install it or provide a .wav file."
            ) from exc

        payload = torch.load(reference_path, map_location="cpu")
        reference_embeddings = _extract_embeddings_from_pt(
            payload,
            reference_path=reference_path,
        )
        print(f"[INFO] reference embedding file: {reference_path}")
        print(f"[INFO] loaded {len(reference_embeddings)} enrolled embedding(s)")
        first_label, first_embedding = reference_embeddings[0]
        print(
            "[INFO] first reference embedding stats: "
            f"entry={first_label} shape={first_embedding.shape} dtype={first_embedding.dtype} "
            f"min={first_embedding.min():.4f} max={first_embedding.max():.4f}"
        )
        return reference_embeddings

    raise ValueError(
        f"reference input must be a .wav or .pt file: {reference_path}"
    )


def _print_io_metadata(session: "ort.InferenceSession") -> None:
    input_meta = session.get_inputs()[0]
    print(f"[INFO] input name: {input_meta.name}")
    print(f"[INFO] input shape: {input_meta.shape}")
    print(f"[INFO] input type: {input_meta.type}")
    for idx, output_meta in enumerate(session.get_outputs()):
        print(f"[INFO] output[{idx}] name: {output_meta.name}")
        print(f"[INFO] output[{idx}] shape: {output_meta.shape}")
        print(f"[INFO] output[{idx}] type: {output_meta.type}")


def main() -> int:
    args = parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency 'onnxruntime'. Install it first, for example: "
            "pip install onnxruntime"
        ) from exc

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    onnx_model = onnx.load(str(model_path))
    onnx.checker.check_model(onnx_model)
    print(f"[OK] ONNX model validated: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=args.providers)
    _print_io_metadata(session)

    input_meta = session.get_inputs()[0]
    expected_shape = input_meta.shape

    input_audio = args.input_audio.resolve()
    reference_audio = args.reference_audio.resolve()

    if not input_audio.exists():
        raise FileNotFoundError(f"input audio not found: {input_audio}")
    if input_audio.suffix.lower() != ".wav":
        raise ValueError(f"input audio must be a .wav file: {input_audio}")
    if not reference_audio.exists():
        raise FileNotFoundError(f"reference input not found: {reference_audio}")

    input_tensor = _coerce_input_shape(
        _preprocess_audio(
            input_audio,
            sr=args.sr,
            n_mels=args.n_mels,
            duration=args.duration,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
        ),
        expected_shape,
    )
    reference_embeddings = _load_reference_embeddings(
        reference_audio,
        session=session,
        input_name=input_meta.name,
        expected_shape=expected_shape,
        args=args,
    )

    print(f"[INFO] input audio: {input_audio}")
    print(
        "[INFO] input tensor stats: "
        f"shape={input_tensor.shape} dtype={input_tensor.dtype} "
        f"min={input_tensor.min():.4f} max={input_tensor.max():.4f}"
    )

    input_embedding = np.asarray(session.run(None, {input_meta.name: input_tensor})[0], dtype=np.float32)

    scored_references = []
    for reference_label, reference_embedding in reference_embeddings:
        cosine_similarity = _cosine_similarity(input_embedding, reference_embedding)
        scored_references.append(
            {
                "label": reference_label,
                "embedding": reference_embedding,
                "similarity": cosine_similarity,
                "distance": 1.0 - cosine_similarity,
            }
        )

    similarities = np.asarray(
        [item["similarity"] for item in scored_references],
        dtype=np.float64,
    )
    log_probs = _log_softmax(similarities)
    probs = np.exp(log_probs)
    for item, log_prob, prob in zip(scored_references, log_probs, probs):
        item["log_prob"] = float(log_prob)
        item["probability"] = float(prob)

    scored_references.sort(key=lambda item: item["similarity"], reverse=True)
    best_match = scored_references[0]
    best_label = best_match["label"]
    best_embedding = best_match["embedding"]
    best_similarity = best_match["similarity"]
    best_distance = best_match["distance"]
    best_log_prob = best_match["log_prob"]
    best_probability = best_match["probability"]
    accepted_match = best_similarity >= args.match_threshold

    print(f"[OK] input embedding shape: {input_embedding.shape}")
    print(f"[OK] compared against {len(scored_references)} reference embedding(s)")
    print(f"[OK] best reference embedding shape: {best_embedding.shape}")
    print(f"[RESULT] best match: {best_label}")
    print(f"[RESULT] best cosine similarity: {best_similarity:.6f}")
    print(f"[RESULT] best cosine distance: {best_distance:.6f}")
    print(f"[RESULT] best log-softmax score: {best_log_prob:.6f}")
    print(f"[RESULT] best probability: {best_probability:.6%}")
    print(f"[RESULT] acceptance threshold: {args.match_threshold:.6f}")
    if accepted_match:
        print(f"[RESULT] decision: accepted as {best_label}")
    else:
        print("[RESULT] decision: not that person")
    if len(scored_references) > 1:
        print("[INFO] top matches:")
        for item in scored_references[:5]:
            print(
                f"  {item['label']}: similarity={item['similarity']:.6f} "
                f"distance={item['distance']:.6f} "
                f"log_prob={item['log_prob']:.6f} "
                f"probability={item['probability']:.6%}"
            )
    print(
        "[INFO] input embedding preview: "
        f"{np.array2string(input_embedding.reshape(-1)[:8], precision=5, separator=', ')}"
    )
    print(
        "[INFO] best reference embedding preview: "
        f"{np.array2string(best_embedding.reshape(-1)[:8], precision=5, separator=', ')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
