#!/bin/bash
# PRM-Guided Decoding for Math with Unified Model
# Uses base model with unified multi-dimensional classifier
# Single model covers all 7 objectives

SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-10}
BASE_GPU_ID=0
REWARD_GPU_ID=1

mkdir -p results 

CUDA_VISIBLE_DEVICES=${BASE_GPU_ID},${REWARD_GPU_ID} python mahdpo_unified_guide_math.py \
  --base_model_name Qwen/Qwen2.5-7B-Instruct \
  --unified_reward_model_name andre930/my_unified_model_classification_all_2108 \
  --use_reward_guidance \
  --candidate_selection unified_guided \
  --num_samples 500 \
  --split_index ${SPLIT_INDEX} \
  --total_splits ${TOTAL_SPLITS} \
  --output_file results/math_unified_split${SPLIT_INDEX}.jsonl \
  --base_device_id 0 \
  --reward_device_id 1 \
  --dtype bfloat16 \
  --cache_dir /tmp/unified_math_cache_split${SPLIT_INDEX} \
  --seed 42
