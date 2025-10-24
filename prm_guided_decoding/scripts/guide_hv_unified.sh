#!/bin/bash
# PRM-Guided Decoding for Human Values with Unified Model
# Uses base model with unified multi-dimensional classifier
# Single model covers all 7 objectives

# Accept split parameters from command line or use defaults
SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-10}
GPU_ID=0

mkdir -p results 

CUDA_VISIBLE_DEVICES=${GPU_ID} python mahdpo_unified_guide_hv.py \
  --total_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_id ${SPLIT_INDEX} \
  --base_model_name meta-llama/Llama-3.1-8B-Instruct \
  --unified_reward_model_name andre930/my_unified_model_classification_all_2108 \
  --use_base_model_only \
  --num_candidates 5 \
  --dtype bfloat16 \
  --cache_dir /tmp/unified_hv_cache_split${SPLIT_INDEX} \
  --output_file results/hv_unified_split${SPLIT_INDEX}.jsonl \
  --unified_device cuda:0 \
  --max_total_tokens 1024
