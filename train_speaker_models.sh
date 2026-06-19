#!/usr/bin/env bash
# Run the local speaker-model training workflows:
#   1. initialize x-vector backbone
#   2. train x-vector with AAM-Softmax, without prototypical network
#   3. train x-vector + prototypical network
#   4. train ECAPA-TDNN with AAM-Softmax, without prototypical network

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

# Dataset selection. Leave DATASET_ROOT empty to let the Python scripts auto-detect
# audio_data/ or LIBRISPEECH_ROOT/train-clean-100.
DATASET_ROOT="${DATASET_ROOT:-"../Nemo_SR/data/speakerdataset/LibriSpeech/train-clean-100"}"
EXT="${EXT:-.flac}"

# Common split/filter settings.
NUM_SPEAKERS="${NUM_SPEAKERS:-250}"
TRAIN_RATIO="${TRAIN_RATIO:-0.6}"
VAL_RATIO="${VAL_RATIO:-0.2}"
TEST_RATIO="${TEST_RATIO:-0.2}"
MIN_SAMPLES_PER_SPEAKER="${MIN_SAMPLES_PER_SPEAKER:-20}"
MAX_SAMPLES_PER_SPEAKER="${MAX_SAMPLES_PER_SPEAKER:-}"
SEED="${SEED:-42}"

# Audio/frontend settings.
SR="${SR:-16000}"
N_MELS="${N_MELS:-80}"
DURATION="${DURATION:-5.0}"
N_FFT="${N_FFT:-512}"
HOP_LENGTH="${HOP_LENGTH:-256}"

# x-vector architecture.
XVECTOR_EMBEDDING_DIM="${XVECTOR_EMBEDDING_DIM:-192}"
XVECTOR_TDNN_CHANNELS="${XVECTOR_TDNN_CHANNELS:-512}"
XVECTOR_STATS_CHANNELS="${XVECTOR_STATS_CHANNELS:-1500}"
XVECTOR_DROPOUT="${XVECTOR_DROPOUT:-0.1}"

# x-vector supervised training without ProtoNet.
XVECTOR_AAM_EPOCHS="${XVECTOR_AAM_EPOCHS:-50}"
XVECTOR_AAM_BATCH_SIZE="${XVECTOR_AAM_BATCH_SIZE:-128}"
XVECTOR_AAM_LR="${XVECTOR_AAM_LR:-1e-4}"
XVECTOR_AAM_WEIGHT_DECAY="${XVECTOR_AAM_WEIGHT_DECAY:-0.01}"
XVECTOR_AAM_NUM_WORKERS="${XVECTOR_AAM_NUM_WORKERS:-0}"
XVECTOR_AAM_MAX_EVAL_PAIRS="${XVECTOR_AAM_MAX_EVAL_PAIRS:-20000}"

# x-vector + ProtoNet training. Defaults mirror the ECAPA ProtoNet config in main.py.
PROTO_N_WAY="${PROTO_N_WAY:-5}"
PROTO_K_SHOT="${PROTO_K_SHOT:-5}"
PROTO_N_QUERY="${PROTO_N_QUERY:-15}"
PROTO_EPISODES="${PROTO_EPISODES:-500}"
PROTO_VAL_EPISODES="${PROTO_VAL_EPISODES:-2}"
PROTO_TEST_EPISODES="${PROTO_TEST_EPISODES:-50}"
PROTO_LOSS_MODE="${PROTO_LOSS_MODE:-hybrid}"
PROTO_SCALE="${PROTO_SCALE:-30.0}"
PROTO_MARGIN="${PROTO_MARGIN:-0.15}"
AAM_SCALE="${AAM_SCALE:-30.0}"
AAM_MARGIN="${AAM_MARGIN:-0.15}"
HYBRID_PROTO_WEIGHT="${HYBRID_PROTO_WEIGHT:-0.8}"
HYBRID_AAM_WEIGHT="${HYBRID_AAM_WEIGHT:-0.2}"
PROTO_LR="${PROTO_LR:-1e-4}"
PROTO_WEIGHT_DECAY="${PROTO_WEIGHT_DECAY:-0.01}"
EVAL_SEED="${EVAL_SEED:-36}"

# ECAPA-TDNN supervised training without ProtoNet.
ECAPA_EPOCHS="${ECAPA_EPOCHS:-50}"
ECAPA_BATCH_SIZE="${ECAPA_BATCH_SIZE:-128}"
ECAPA_LR="${ECAPA_LR:-1e-4}"
ECAPA_WEIGHT_DECAY="${ECAPA_WEIGHT_DECAY:-0.01}"
ECAPA_CHANNELS="${ECAPA_CHANNELS:-512}"
ECAPA_EMBEDDING_DIM="${ECAPA_EMBEDDING_DIM:-192}"
ECAPA_NUM_WORKERS="${ECAPA_NUM_WORKERS:-0}"
ECAPA_MAX_EVAL_PAIRS="${ECAPA_MAX_EVAL_PAIRS:-20000}"

# Augmentation/VAD.
TRAIN_AUGMENT="${TRAIN_AUGMENT:-1}"
AUGMENTATION_PROBABILITY="${AUGMENTATION_PROBABILITY:-0.2}"
AUGMENTATION_RIR_DIR="${AUGMENTATION_RIR_DIR:-$ROOT_DIR/rirs_noises/RIRS_NOISES/real_rirs_isotropic_noises}"
VAD_ENABLED="${VAD_ENABLED:-1}"
VAD_TOP_DB="${VAD_TOP_DB:-10.0}"
VAD_FRAME_LENGTH="${VAD_FRAME_LENGTH:-2048}"
VAD_HOP_LENGTH="${VAD_HOP_LENGTH:-258}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
ALLOW_CPU="${ALLOW_CPU:-0}"

# Stage toggles.
RUN_XVECTOR_INIT="${RUN_XVECTOR_INIT:-1}"
RUN_XVECTOR_NO_PROTONET="${RUN_XVECTOR_NO_PROTONET:-1}"
RUN_XVECTOR_PROTONET="${RUN_XVECTOR_PROTONET:-1}"
RUN_ECAPA_NO_PROTONET="${RUN_ECAPA_NO_PROTONET:-1}"
DRY_RUN="${DRY_RUN:-0}"

XVECTOR_INIT_CHECKPOINT="${XVECTOR_INIT_CHECKPOINT:-$OUTPUT_DIR/xvector_init.pth}"
XVECTOR_AAM_MODEL="${XVECTOR_AAM_MODEL:-$OUTPUT_DIR/xvector_aam_model.pth}"
XVECTOR_AAM_PLOT="${XVECTOR_AAM_PLOT:-$OUTPUT_DIR/xvector_aam_curves.png}"
XVECTOR_AAM_DET="${XVECTOR_AAM_DET:-$OUTPUT_DIR/xvector_aam_det_curve.png}"
XVECTOR_PROTONET_MODEL="${XVECTOR_PROTONET_MODEL:-$OUTPUT_DIR/xvector_protonet_model.pth}"
XVECTOR_PROTONET_PLOT="${XVECTOR_PROTONET_PLOT:-$OUTPUT_DIR/xvector_protonet_curves.png}"
XVECTOR_PROTONET_DET="${XVECTOR_PROTONET_DET:-$OUTPUT_DIR/xvector_protonet_det_curve.png}"
ECAPA_MODEL="${ECAPA_MODEL:-$OUTPUT_DIR/ecapa_tdnn_aam_model.pth}"
ECAPA_PLOT="${ECAPA_PLOT:-$OUTPUT_DIR/ecapa_tdnn_aam_curves.png}"
ECAPA_DET="${ECAPA_DET:-$OUTPUT_DIR/ecapa_tdnn_aam_det_curve.png}"

mkdir -p "$OUTPUT_DIR"

bool_flag() {
    local value="$1"
    [[ "$value" == "1" || "$value" == "true" || "$value" == "TRUE" || "$value" == "yes" || "$value" == "YES" ]]
}

run_cmd() {
    echo ""
    echo "+ $*"
    if bool_flag "$DRY_RUN"; then
        return 0
    fi
    "$@"
}

add_dataset_args() {
    local -n out_args="$1"
    if [[ -n "$DATASET_ROOT" ]]; then
        out_args+=(--dataset-root "$DATASET_ROOT")
    fi
    if [[ -n "$EXT" ]]; then
        out_args+=(--ext "$EXT")
    fi
}

add_common_dataset_args() {
    local -n out_args="$1"
    out_args+=(
        --num-speakers "$NUM_SPEAKERS"
        --train-ratio "$TRAIN_RATIO"
        --val-ratio "$VAL_RATIO"
        --test-ratio "$TEST_RATIO"
        --min-samples-per-speaker "$MIN_SAMPLES_PER_SPEAKER"
        --sr "$SR"
        --n-mels "$N_MELS"
        --duration "$DURATION"
        --n-fft "$N_FFT"
        --hop-length "$HOP_LENGTH"
        --seed "$SEED"
    )
    if [[ -n "$MAX_SAMPLES_PER_SPEAKER" ]]; then
        out_args+=(--max-samples-per-speaker "$MAX_SAMPLES_PER_SPEAKER")
    fi
    if bool_flag "$VAD_ENABLED"; then
        out_args+=(
            --vad-enabled
            --vad-top-db "$VAD_TOP_DB"
            --vad-frame-length "$VAD_FRAME_LENGTH"
            --vad-hop-length "$VAD_HOP_LENGTH"
        )
    fi
    if ! bool_flag "$TRAIN_AUGMENT"; then
        out_args+=(--no-train-augment)
    fi
    if ! bool_flag "$SHOW_PROGRESS"; then
        out_args+=(--no-progress)
    fi
    if bool_flag "$ALLOW_CPU"; then
        out_args+=(--allow-cpu)
    fi
}

echo "======================================"
echo "Speaker Model Training Automation"
echo "======================================"
echo "Python: $($PYTHON_BIN --version)"
echo "Project: $ROOT_DIR"
echo "Output: $OUTPUT_DIR"
if [[ -n "$DATASET_ROOT" ]]; then
    echo "Dataset root: $DATASET_ROOT"
else
    echo "Dataset root: auto-detect"
fi
if [[ -n "$EXT" ]]; then
    echo "Audio extension: $EXT"
else
    echo "Audio extension: auto-detect"
fi

if bool_flag "$RUN_XVECTOR_INIT"; then
    run_cmd "$PYTHON_BIN" init_xvector_model.py \
        --output "$XVECTOR_INIT_CHECKPOINT" \
        --n-mels "$N_MELS" \
        --tdnn-channels "$XVECTOR_TDNN_CHANNELS" \
        --stats-channels "$XVECTOR_STATS_CHANNELS" \
        --embedding-dim "$XVECTOR_EMBEDDING_DIM" \
        --dropout "$XVECTOR_DROPOUT" \
        --seed "$SEED"
fi

if bool_flag "$RUN_XVECTOR_NO_PROTONET"; then
    echo ""
    echo "=== STAGE: x-vector supervised, no ProtoNet ==="
    xvector_aam_args=()
    add_dataset_args xvector_aam_args
    add_common_dataset_args xvector_aam_args
    xvector_aam_args+=(
        --epochs "$XVECTOR_AAM_EPOCHS"
        --batch-size "$XVECTOR_AAM_BATCH_SIZE"
        --lr "$XVECTOR_AAM_LR"
        --weight-decay "$XVECTOR_AAM_WEIGHT_DECAY"
        --tdnn-channels "$XVECTOR_TDNN_CHANNELS"
        --stats-channels "$XVECTOR_STATS_CHANNELS"
        --embedding-dim "$XVECTOR_EMBEDDING_DIM"
        --dropout "$XVECTOR_DROPOUT"
        --aam-scale "$AAM_SCALE"
        --aam-margin "$AAM_MARGIN"
        --num-workers "$XVECTOR_AAM_NUM_WORKERS"
        --max-eval-pairs "$XVECTOR_AAM_MAX_EVAL_PAIRS"
        --output-model "$XVECTOR_AAM_MODEL"
        --plot-path "$XVECTOR_AAM_PLOT"
        --det-curve-path "$XVECTOR_AAM_DET"
        --augmentation-probability "$AUGMENTATION_PROBABILITY"
        --augmentation-rir-dir "$AUGMENTATION_RIR_DIR"
    )
    if [[ -f "$XVECTOR_INIT_CHECKPOINT" ]] || bool_flag "$RUN_XVECTOR_INIT"; then
        xvector_aam_args+=(--checkpoint "$XVECTOR_INIT_CHECKPOINT")
    fi
    run_cmd "$PYTHON_BIN" train_xvector_no_protonet.py "${xvector_aam_args[@]}"
fi

if bool_flag "$RUN_XVECTOR_PROTONET"; then
    xvector_args=()
    add_dataset_args xvector_args
    add_common_dataset_args xvector_args
    if ! bool_flag "$VAD_ENABLED"; then
        xvector_args+=(--no-vad)
    fi
    xvector_args+=(
        --n-way "$PROTO_N_WAY"
        --k-shot "$PROTO_K_SHOT"
        --n-query "$PROTO_N_QUERY"
        --episodes "$PROTO_EPISODES"
        --val-episodes "$PROTO_VAL_EPISODES"
        --embedding-dim "$XVECTOR_EMBEDDING_DIM"
        --tdnn-channels "$XVECTOR_TDNN_CHANNELS"
        --stats-channels "$XVECTOR_STATS_CHANNELS"
        --dropout "$XVECTOR_DROPOUT"
        --training-loss-mode "$PROTO_LOSS_MODE"
        --proto-scale "$PROTO_SCALE"
        --proto-margin "$PROTO_MARGIN"
        --aam-scale "$AAM_SCALE"
        --aam-margin "$AAM_MARGIN"
        --hybrid-proto-weight "$HYBRID_PROTO_WEIGHT"
        --hybrid-aam-weight "$HYBRID_AAM_WEIGHT"
        --lr "$PROTO_LR"
        --weight-decay "$PROTO_WEIGHT_DECAY"
        --output-model "$XVECTOR_PROTONET_MODEL"
        --plot-path "$XVECTOR_PROTONET_PLOT"
        --det-curve-path "$XVECTOR_PROTONET_DET"
        --augmentation-probability "$AUGMENTATION_PROBABILITY"
        --augmentation-rir-dir "$AUGMENTATION_RIR_DIR"
        --eval-seed "$EVAL_SEED"
    )
    if [[ -n "$PROTO_TEST_EPISODES" ]]; then
        xvector_args+=(--test-episodes "$PROTO_TEST_EPISODES")
    fi
    if [[ -f "$XVECTOR_INIT_CHECKPOINT" ]] || bool_flag "$RUN_XVECTOR_INIT"; then
        xvector_args+=(--checkpoint "$XVECTOR_INIT_CHECKPOINT")
    fi
    run_cmd "$PYTHON_BIN" train_xvector_protonet.py "${xvector_args[@]}"
fi

if bool_flag "$RUN_ECAPA_NO_PROTONET"; then
    ecapa_args=()
    add_dataset_args ecapa_args
    add_common_dataset_args ecapa_args
    ecapa_args+=(
        --epochs "$ECAPA_EPOCHS"
        --batch-size "$ECAPA_BATCH_SIZE"
        --lr "$ECAPA_LR"
        --weight-decay "$ECAPA_WEIGHT_DECAY"
        --channels "$ECAPA_CHANNELS"
        --embedding-dim "$ECAPA_EMBEDDING_DIM"
        --aam-scale "$AAM_SCALE"
        --aam-margin "$AAM_MARGIN"
        --num-workers "$ECAPA_NUM_WORKERS"
        --max-eval-pairs "$ECAPA_MAX_EVAL_PAIRS"
        --output-model "$ECAPA_MODEL"
        --plot-path "$ECAPA_PLOT"
        --det-curve-path "$ECAPA_DET"
        --augmentation-probability "$AUGMENTATION_PROBABILITY"
        --augmentation-rir-dir "$AUGMENTATION_RIR_DIR"
    )
    run_cmd "$PYTHON_BIN" train_ecapa_tdnn_no_protonet.py "${ecapa_args[@]}"
fi

echo ""
echo "Done."
echo "Artifacts:"
echo "  x-vector init:        $XVECTOR_INIT_CHECKPOINT"
echo "  x-vector no ProtoNet: $XVECTOR_AAM_MODEL"
echo "  x-vector + ProtoNet:  $XVECTOR_PROTONET_MODEL"
echo "  ECAPA no ProtoNet:    $ECAPA_MODEL"
