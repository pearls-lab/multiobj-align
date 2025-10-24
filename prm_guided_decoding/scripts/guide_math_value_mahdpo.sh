#!/bin/bash
# PRM-Guided Decoding for Math (MAH-DPO Model + Value Model)
# Uses MAH-DPO checkpoint with Value Model guidance
# Multi-GPU setup: base on GPU 0, value model on GPU 1

SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-4}
BASE_GPU_ID=0
VALUE_GPU_ID=1

mkdir -p results 

# Update checkpoint path to your trained MAH-DPO model
CHECKPOINT_PATH="Jennny/good_sft_mdpo_step-LATEST"

CUDA_VISIBLE_DEVICES=${BASE_GPU_ID},${VALUE_GPU_ID} python mahdpo_math_acc_guide.py \
  --base_model_name Qwen/Qwen2.5-7B-Instruct \
  --dpo_checkpoint_path ${CHECKPOINT_PATH} \
  --value_model_name Jennny/qwen-math-value-model-join-09-5e5-3ep \
  --num_heads 2 \
  --num_samples 500 \
  --split_index ${SPLIT_INDEX} \
  --total_splits ${TOTAL_SPLITS} \
  --output_file results/math_mdpo_value_split${SPLIT_INDEX}.jsonl \
  --base_device_id 0 \
  --value_device_id 1 \
  --dtype bfloat16 \
  --cache_dir /tmp/math_mdpo_cache_split${SPLIT_INDEX} \
  --seed 42
