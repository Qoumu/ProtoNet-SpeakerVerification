# Quick Start

This project supports four practical workflows:

1. Train an ECAPA-TDNN prototypical-network checkpoint.
2. Enroll speakers into an embedding store.
3. Export the model to ONNX or RKNN.
4. Sanity-check an exported ONNX model against a `.wav` or `.pt` reference.

## Project Outputs

The main artifacts used across the project are:

- `output/ECAPATDNN_protonet_model.pth`: trained checkpoint
- `output/ECAPATDNN_protonet_curves.png`: training curves
- `output/ECAPATDNN_protonet_det_curve.png`: DET / EER plot
- `output/enrolled_speakers.pt`: default speaker enrollment store
- `output/enrolled_speakers.npy`: optional legacy embedding matrix
- `output/enrolled_speakers_ids.txt`: speaker IDs for the legacy `.npy` store
- `output/ecapa_tdnn.onnx`: exported ONNX model
- `output/ecapa_tdnn.rknn`: exported RKNN model

## 1. Set Up Python

Minimum host dependencies for enrollment:

```bash
pip install -r requirements_enrollment.txt
```

Additional packages are needed for the other workflows:

- training: `matplotlib`, `torchaudio`, `tqdm`
- ONNX export and validation: `onnx`, `onnxscript`, `onnxruntime`
- RKNN export: `rknn-toolkit2` in a Rockchip-supported Python environment
- calibration helper: `scipy`

Example:

```bash
pip install matplotlib torchaudio tqdm onnx onnxscript onnxruntime scipy
```

## 2. Train a Checkpoint

`main.py` trains on LibriSpeech `train-clean-100`. The dataset root is resolved in this order:

1. `LIBRISPEECH_ROOT`
2. `data/speakerdataset/LibriSpeech` under this repo
3. `../data/speakerdataset/LibriSpeech`
4. `../Nemo_SR/data/speakerdataset/LibriSpeech`

Set the dataset path explicitly if you do not use one of those layouts:

```bash
export LIBRISPEECH_ROOT=/absolute/path/to/LibriSpeech
python3 main.py
```

Notes:

- `main.py` expects audio under `$LIBRISPEECH_ROOT/train-clean-100`
- the default training split uses 250 speakers
- speakers with fewer than 15 clips are excluded
- the checkpoint is written to `output/ECAPATDNN_protonet_model.pth`

## 3. Enroll Speakers

Use the trained checkpoint to build a speaker store. The examples below write artifacts into `output/` so they line up with the rest of the repo.

```bash
python3 enroll_lite.py \
    --model-path output/ECAPATDNN_protonet_model.pth \
    --speaker-id alice \
    --audio-files audio_samples/alice/*.wav \
    --store-path output/enrolled_speakers.pt \
    --verbose

python3 enroll_lite.py \
    --model-path output/ECAPATDNN_protonet_model.pth \
    --speaker-id bob \
    --audio-files audio_samples/bob/*.wav \
    --store-path output/enrolled_speakers.pt \
    --verbose
```

Behavior:

- enrolling a new speaker appends a new entry
- re-enrolling an existing speaker updates that speaker in place
- `.pt` is the default format and stores embeddings by speaker ID

Inspect a `.pt` store:

```bash
python3 - <<'PY'
import torch
store = torch.load("output/enrolled_speakers.pt", map_location="cpu")
print("speakers:", list(store))
first = next(iter(store.values()))
print("embedding shape:", tuple(first.shape))
print("dtype:", first.dtype)
PY
```

If your downstream runtime expects a flat matrix instead of a PyTorch store, use the legacy `.npy` format:

```bash
python3 enroll_lite.py \
    --model-path output/ECAPATDNN_protonet_model.pth \
    --speaker-id alice \
    --audio-files audio_samples/alice/*.wav \
    --store-path output/enrolled_speakers.npy
```

That also writes `output/enrolled_speakers_ids.txt`.

## 4. Export ONNX

If you want an ONNX model for host-side validation, export only the ONNX step first:

```bash
python3 export_to_rknn.py \
    --checkpoint output/ECAPATDNN_protonet_model.pth \
    --onnx-out output/ecapa_tdnn.onnx \
    --onnx-only
```

Important:

- `main.py` saves the checkpoint under `output/`
- `export_to_rknn.py` defaults to `ECAPATDNN_protonet_model.pth` in the repo root
- pass `--checkpoint output/ECAPATDNN_protonet_model.pth` unless you copied the file elsewhere

## 5. Validate ONNX

`test_onnx.py` compares a query `.wav` against either:

- another `.wav`
- an enrolled `.pt` speaker store

Example using an enrollment store:

```bash
python3 test_onnx.py \
    --model output/ecapa_tdnn.onnx \
    --input-audio path/to/query.wav \
    --reference-audio output/enrolled_speakers.pt
```

Example using a single reference wave file:

```bash
python3 test_onnx.py \
    --model output/ecapa_tdnn.onnx \
    --input-audio path/to/query.wav \
    --reference-audio path/to/reference.wav
```

## 6. Export RKNN

Basic RKNN export:

```bash
python3 export_to_rknn.py \
    --checkpoint output/ECAPATDNN_protonet_model.pth \
    --onnx-out output/ecapa_tdnn.onnx \
    --rknn-out output/ecapa_tdnn.rknn \
    --target-platform rk3566
```

For `rv1103` and `rv1106`, RKNN export requires INT8 quantization. Use real calibration audio when possible:

```bash
python3 export_to_rknn.py \
    --checkpoint output/ECAPATDNN_protonet_model.pth \
    --onnx-out output/ecapa_tdnn.onnx \
    --rknn-out output/ecapa_tdnn.rknn \
    --target-platform rv1106 \
    --quantize \
    --dataset /path/to/calibration_audio_dir
```

`--dataset` accepts any of these:

- a directory of audio files
- a single audio file
- a text file listing audio files
- an RKNN `dataset.txt` containing `.npy` calibration tensors

If you only need a quick build and accept lower-quality calibration, you can allow random inputs:

```bash
python3 export_to_rknn.py \
    --checkpoint output/ECAPATDNN_protonet_model.pth \
    --onnx-out output/ecapa_tdnn.onnx \
    --rknn-out output/ecapa_tdnn.rknn \
    --target-platform rv1106 \
    --allow-random-calib
```

## 7. Build a Calibration Dataset Ahead of Time

If you prefer to generate `.npy` calibration tensors first:

```bash
python3 prepare_rknn_calibration_dataset.py \
    --audio-list dataset_absolute.txt \
    --output-dir rknn_calibration_librispeech_100
```

Then pass the generated dataset file to RKNN export:

```bash
python3 export_to_rknn.py \
    --checkpoint output/ECAPATDNN_protonet_model.pth \
    --onnx-out output/ecapa_tdnn.onnx \
    --rknn-out output/ecapa_tdnn.rknn \
    --target-platform rv1106 \
    --quantize \
    --dataset rknn_calibration_librispeech_100/dataset.txt
```

## Common Gotchas

- `enroll_lite.py` and `export_to_rknn.py` default to `ECAPATDNN_protonet_model.pth` in the repo root, but `main.py` writes the trained model to `output/ECAPATDNN_protonet_model.pth`.
- `test_onnx.py` defaults to `output/ecapa_tdnn.onnx`, so keep the ONNX export path aligned or pass `--model`.
- `test_onnx.py` requires `--input-audio` to be a `.wav` file.
