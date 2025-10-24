#!/bin/bash
# Inference for Socratic Mind with multiple configurations
# Each configuration is for a different task - run separately or allocate different GPUs for different splits
# Supports: base model, single head (accuracy/engagement), ensemble, and SFT baseline

# Update these paths
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DPO_CHECKPOINT="Jennny/sm_mdpo_qwen25_fsdp_5e6_2ep_bz128_LATEST_with_SFT_updated"
SFT_CHECKPOINT="Jennny/sm_sft_qwen25_fsdp_5e6_2ep_bz128_LATEST_updated"
DATA_FILE="data/sm_acc.json"
SPLIT_INDEX=0
TOTAL_SPLITS=5
GPU_ID=0

SYSTEM_PROMPT="You are a tutor who is helping a beginner student learn programming. Continue as the same tutor and reply similarly to the last student message, matching EXACTLY the SAME speaking tone and tutoring style as in your earlier messages (e.g. reply to the student's last message concisely in 1-2 sentences and then always ask a meaningful follow-up question)."

# Base Model
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_sm.py \
  --model_type base \
  --model_name ${BASE_MODEL} \
  --data_file ${DATA_FILE} \
  --num_samples 100 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --max_new_tokens 300 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --output_file results/base_qwen25_sm_split${SPLIT_INDEX}.jsonl \
  --system_prompt "${SYSTEM_PROMPT}"

# SFT Baseline
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_sm.py \
  --model_type sft \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${SFT_CHECKPOINT} \
  --data_file ${DATA_FILE} \
  --num_samples 100 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --max_new_tokens 300 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --output_file results/sm_sft_split${SPLIT_INDEX}.jsonl \
  --system_prompt "${SYSTEM_PROMPT}"

# Head 0 (Accuracy) - For Head 1 (Engagement), use --head_index 1
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_sm.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 2 \
  --head_index 0 \
  --data_file ${DATA_FILE} \
  --num_samples 100 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --max_new_tokens 300 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --output_file results/sm_head0_split${SPLIT_INDEX}.jsonl \
  --system_prompt "${SYSTEM_PROMPT}"

# Ensemble (both heads)
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_sm.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 2 \
  --use_ensemble \
  --data_file ${DATA_FILE} \
  --num_samples 100 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --max_new_tokens 300 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --output_file results/sm_ensemble_split${SPLIT_INDEX}.jsonl \
  --system_prompt "${SYSTEM_PROMPT}"

