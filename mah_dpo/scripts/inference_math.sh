#!/bin/bash
# Inference for Math with multiple configurations
# Each configuration is for a different task - run separately or allocate different GPUs for different splits
# Supports: base model, SFT, single head, and various ensemble configurations

# Update these paths
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DPO_CHECKPOINT="Jennny/good_sft_mdpo_step-LATEST"
DPO_CHECKPOINT_EARLY="Jennny/good_sft_mdpo_step-9984"
SFT_CHECKPOINT="Jennny/single_dpo_math_good_sft_5e6_2ep_step_4992"
SPLIT_INDEX=0
TOTAL_SPLITS=5
GPU_ID=0

# Base Model
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_math.py \
  --model_type base \
  --model_name ${BASE_MODEL} \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/math_qwen7b_base_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 2048

# SFT Baseline
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_math.py \
  --model_type sft \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${SFT_CHECKPOINT} \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/math_sft_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 2048

# Head 0 (Accuracy) - For Head 1 (Engagement), use --head_index 1 and ${DPO_CHECKPOINT}
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_math.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT_EARLY} \
  --num_heads 2 \
  --head_index 0 \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/math_head0_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 2048

# Ensemble - Equal Weights (0.5, 0.5)
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_math.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 2 \
  --use_ensemble \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/math_ensemble_equal_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 2048

# Ensemble - Accuracy Focused (0.75, 0.25)
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_math.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 2 \
  --use_ensemble \
  --head_weights 0.75 0.25 \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/math_ensemble_acc_focused_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 2048

# Ensemble - Engagement Focused (0.25, 0.75)
CUDA_VISIBLE_DEVICES=${GPU_ID} python inference_math.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 2 \
  --use_ensemble \
  --head_weights 0.25 0.75 \
  --num_samples 500 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --output_file results/math_ensemble_eng_focused_split${SPLIT_INDEX}.jsonl \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --max_new_tokens 2048
