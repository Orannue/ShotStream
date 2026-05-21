#!/usr/bin/env bash
set -euo pipefail

JSON_DIR=${JSON_DIR:-demo/testdata/eval_caption_multishot_t2v_100_json}
CSV_PATH=${CSV_PATH:-demo/testdata/eval_caption_multishot_t2v_100.csv}
OUTPUT_DIR=${OUTPUT_DIR:-demo/infer/eval_caption_multishot_t2v_100}
FRAMES_PER_SHOT=${FRAMES_PER_SHOT:-81}
CONFIG_PATH=${CONFIG_PATH:-ckpts/shotstream.yaml}
DEFAULT_CONFIG_PATH=${DEFAULT_CONFIG_PATH:-ckpts/default_config.yaml}
RESUME_CKPT=${RESUME_CKPT:-ckpts/shotstream_merged.pt}
MODEL_ROOT=${MODEL_ROOT:-..models/Wan2.1-T2V-1.3B}

# python tools/inference/build_multishot_json_csv.py \
#     --json_dir "${JSON_DIR}" \
#     --output_csv "${CSV_PATH}" \
#     --frames_per_shot "${FRAMES_PER_SHOT}"


python Inference_Causal_BatchJson.py \
    --config_path "${CONFIG_PATH}" \
    --default_config_path "${DEFAULT_CONFIG_PATH}" \
    --output_folder "${OUTPUT_DIR}" \
    --resume_ckpt "${RESUME_CKPT}" \
    --model_root "${MODEL_ROOT}" \
    --multi_caption True \
    --frames_per_shot "${FRAMES_PER_SHOT}" \
    --data_path "${CSV_PATH}"
