# Host Enrollment and RKNN Export

The current repo workflow is:

1. Train or provide `output/ECAPATDNN_protonet_model.pth`
2. Enroll speakers with `enroll_lite.py`
3. Export ONNX or RKNN with `export_to_rknn.py`
4. Optionally validate ONNX with `test_onnx.py`

Use [QUICKSTART.md](./QUICKSTART.md) for the full end-to-end guide.

## Minimal Commands

Install the base host dependencies:

```bash
pip install -r requirements_enrollment.txt
```

Enroll speakers:

```bash
python3 enroll_lite.py \
    --model-path output/ECAPATDNN_protonet_model.pth \
    --speaker-id alice \
    --audio-files audio_samples/alice/*.wav \
    --store-path output/enrolled_speakers.pt \
    --verbose
```

Export ONNX only:

```bash
python3 export_to_rknn.py \
    --checkpoint output/ECAPATDNN_protonet_model.pth \
    --onnx-out output/ecapa_tdnn.onnx \
    --onnx-only
```

Export RKNN for `rv1106` with quantization:

```bash
python3 export_to_rknn.py \
    --checkpoint output/ECAPATDNN_protonet_model.pth \
    --onnx-out output/ecapa_tdnn.onnx \
    --rknn-out output/ecapa_tdnn.rknn \
    --target-platform rv1106 \
    --quantize \
    --dataset /path/to/calibration_audio_dir
```

Validate ONNX against an enrolled store:

```bash
python3 test_onnx.py \
    --model output/ecapa_tdnn.onnx \
    --input-audio path/to/query.wav \
    --reference-audio output/enrolled_speakers.pt
```

## Output Formats

- `output/enrolled_speakers.pt`: default enrollment store keyed by speaker ID
- `output/enrolled_speakers.npy`: legacy embedding matrix for C/C++
- `output/enrolled_speakers_ids.txt`: row-order IDs for the legacy `.npy` store
- `output/ecapa_tdnn.onnx`: ONNX export
- `output/ecapa_tdnn.rknn`: RKNN export
