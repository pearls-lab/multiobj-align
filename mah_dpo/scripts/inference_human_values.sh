#!/bin/bash
# Inference for Human Values with multiple configurations
# Each configuration is for a different task - run separately or allocate different GPUs for different splits
# Supports: base model, SFT, single head (helpfulness/honesty/truthfulness), and various ensemble configurations

# Update these paths
BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
DPO_CHECKPOINT="Jennny/new_sft_mdpo_llama8b_3heads_help_hon_truth5e7_1ep_bz120_step19920"
SFT_CHECKPOINT="Jennny/clean_len_human_value_sft_llama8b_5e7_1ep_192_step-4992"
SPLIT_INDEX=0
TOTAL_SPLITS=5
GPU_ID=0

# Base Model
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_hv.py \
  --model_type base \
  --model_name ${BASE_MODEL} \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/hv_llama8b_base_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_new_tokens 1024

# SFT Baseline
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_hv.py \
  --model_type sft \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${SFT_CHECKPOINT} \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/hv_sft_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_new_tokens 1024

# Head 0 (Helpfulness) - For Head 1 (Honesty) or Head 2 (Truthfulness), use --head_index 1 or 2
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_hv.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 3 \
  --head_index 0 \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/hv_head0_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 0 \
  --max_new_tokens 1024

# Ensemble - Helpfulness & Honesty Focused (0.5, 0.5, 0)
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_hv.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 3 \
  --use_ensemble \
  --head_weights 0.5 0.5 0 \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/hv_ensemble_help_hon_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 1024

# Ensemble - Honesty & Truthfulness Focused (0, 0.5, 0.5)
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_hv.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 3 \
  --use_ensemble \
  --head_weights 0 0.5 0.5 \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/hv_ensemble_hon_truth_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 1024
