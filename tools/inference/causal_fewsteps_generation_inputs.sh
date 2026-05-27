#!/usr/bin/env bash
set -euo pipefail

# Generate videos from ShotStream-format JSON files that contain per-shot
# frames_per_shot lists. Each frame count is normalized to the nearest 4n+1
# when the CSV is built.

JSON_DIR=${JSON_DIR:-demo/data/generation_inputs_video_only_json}
CSV_PATH=${CSV_PATH:-demo/data/generation_inputs_video_only.csv}
OUTPUT_DIR=${OUTPUT_DIR:-demo/infer/generation_inputs_video_only}

# Used only when a JSON file does not contain frames_per_shot.
FRAMES_PER_SHOT=${FRAMES_PER_SHOT:-81}

CONFIG_PATH=${CONFIG_PATH:-shotstream.yaml}
DEFAULT_CONFIG_PATH=${DEFAULT_CONFIG_PATH:-default_config.yaml}

HUGGINGFACE_WAN_MODEL_ROOT=${HUGGINGFACE_WAN_MODEL_ROOT:-../models/Wan2.1-T2V-1.3B}
HUGGINGFACE_SHOTSTREAM_CKPT_ROOT=${HUGGINGFACE_SHOTSTREAM_CKPT_ROOT:-ckpts}

RESUME_CKPT=${RESUME_CKPT:-${HUGGINGFACE_SHOTSTREAM_CKPT_ROOT}/shotstream_merged.pt}
MODEL_ROOT=${MODEL_ROOT:-${HUGGINGFACE_WAN_MODEL_ROOT}}

GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NUM_GPUS=${NUM_GPUS:-1}
REBUILD_CSV=${REBUILD_CSV:-1}
MULTI_CAPTION=${MULTI_CAPTION:-True}
USE_WO_ROPE_CACHE=${USE_WO_ROPE_CACHE:-False}
SEED=${SEED:-0}

if [[ "${REBUILD_CSV}" == "1" || "${REBUILD_CSV}" == "true" || "${REBUILD_CSV}" == "True" ]]; then
    python tools/inference/build_multishot_json_csv.py \
        --json_dir "${JSON_DIR}" \
        --output_csv "${CSV_PATH}" \
        --frames_per_shot "${FRAMES_PER_SHOT}"
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" torchrun --nproc_per_node="${NUM_GPUS}" Inference_Causal_BatchJson.py \
        --config_path "${CONFIG_PATH}" \
        --default_config_path "${DEFAULT_CONFIG_PATH}" \
        --output_folder "${OUTPUT_DIR}" \
        --resume_ckpt "${RESUME_CKPT}" \
        --model_root "${MODEL_ROOT}" \
        --multi_caption "${MULTI_CAPTION}" \
        --use_wo_rope_cache "${USE_WO_ROPE_CACHE}" \
        --seed "${SEED}" \
        --frames_per_shot "${FRAMES_PER_SHOT}" \
        --data_path "${CSV_PATH}"
else
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python Inference_Causal_BatchJson.py \
        --config_path "${CONFIG_PATH}" \
        --default_config_path "${DEFAULT_CONFIG_PATH}" \
        --output_folder "${OUTPUT_DIR}" \
        --resume_ckpt "${RESUME_CKPT}" \
        --model_root "${MODEL_ROOT}" \
        --multi_caption "${MULTI_CAPTION}" \
        --use_wo_rope_cache "${USE_WO_ROPE_CACHE}" \
        --seed "${SEED}" \
        --frames_per_shot "${FRAMES_PER_SHOT}" \
        --data_path "${CSV_PATH}"
fi
