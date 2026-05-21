#!/usr/bin/env bash
set -euo pipefail

JSON_DIR=${JSON_DIR:-demo/testdata/eval_caption_multishot_t2v_100_json}
CSV_PATH=${CSV_PATH:-demo/testdata/eval_caption_multishot_t2v_100.csv}
OUTPUT_DIR=${OUTPUT_DIR:-demo/infer/eval_caption_multishot_t2v_100}
FRAMES_PER_SHOT=${FRAMES_PER_SHOT:-81}
NUM_GPUS=${NUM_GPUS:-4}
CONFIG_PATH=${CONFIG_PATH:-shotstream.yaml}
DEFAULT_CONFIG_PATH=${DEFAULT_CONFIG_PATH:-default_config.yaml}

# Local Hugging Face repo paths. These match tools/setup/download_ckpt.sh:
#   git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B wan_models
#   git clone https://huggingface.co/KlingTeam/ShotStream ckpts
HUGGINGFACE_WAN_MODEL_ROOT=${HUGGINGFACE_WAN_MODEL_ROOT:-../models/Wan2.1-T2V-1.3B }
HUGGINGFACE_SHOTSTREAM_CKPT_ROOT=${HUGGINGFACE_SHOTSTREAM_CKPT_ROOT:-ckpts}

RESUME_CKPT=${RESUME_CKPT:-${HUGGINGFACE_SHOTSTREAM_CKPT_ROOT}/shotstream_merged.pt}
MODEL_ROOT=${MODEL_ROOT:-${HUGGINGFACE_WAN_MODEL_ROOT}}

# Optional Hugging Face Hub upload target. Leave HF_UPLOAD_REPO_ID empty to skip upload.
# Authentication is read by huggingface-cli from HF_TOKEN or an existing login.
HF_UPLOAD_REPO_ID=${HF_UPLOAD_REPO_ID:-Orannue/Baseline_results}
HF_UPLOAD_REPO_TYPE=${HF_UPLOAD_REPO_TYPE:-dataset}
HF_UPLOAD_LOCAL_PATH=${HF_UPLOAD_LOCAL_PATH:-${OUTPUT_DIR}}
HF_UPLOAD_PATH=${HF_UPLOAD_PATH:-eval_caption_multishot_t2v_100/shotstream}

# python tools/inference/build_multishot_json_csv.py \
#     --json_dir "${JSON_DIR}" \
#     --output_csv "${CSV_PATH}" \
#     --frames_per_shot "${FRAMES_PER_SHOT}"


if [[ "${NUM_GPUS}" -gt 1 ]]; then
    torchrun --nproc_per_node="${NUM_GPUS}" Inference_Causal_BatchJson.py \
        --config_path "${CONFIG_PATH}" \
        --default_config_path "${DEFAULT_CONFIG_PATH}" \
        --output_folder "${OUTPUT_DIR}" \
        --resume_ckpt "${RESUME_CKPT}" \
        --model_root "${MODEL_ROOT}" \
        --multi_caption True \
        --frames_per_shot "${FRAMES_PER_SHOT}" \
        --data_path "${CSV_PATH}"
else
    python Inference_Causal_BatchJson.py \
        --config_path "${CONFIG_PATH}" \
        --default_config_path "${DEFAULT_CONFIG_PATH}" \
        --output_folder "${OUTPUT_DIR}" \
        --resume_ckpt "${RESUME_CKPT}" \
        --model_root "${MODEL_ROOT}" \
        --multi_caption True \
        --frames_per_shot "${FRAMES_PER_SHOT}" \
        --data_path "${CSV_PATH}"
fi

if [[ -n "${HF_UPLOAD_REPO_ID}" ]]; then
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli was not found. Install huggingface_hub or login before uploading." >&2
        exit 1
    fi

    huggingface-cli upload \
        "${HF_UPLOAD_REPO_ID}" \
        "${HF_UPLOAD_LOCAL_PATH}" \
        "${HF_UPLOAD_PATH}" \
        --repo-type "${HF_UPLOAD_REPO_TYPE}"
fi
