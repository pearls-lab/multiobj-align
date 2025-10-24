#!/bin/bash
# PRM-Guided Decoding for Human Values (Base Model + ORM)
# Uses Llama-3.1-8B-Instruct with Helpfulness ORM guidance
# 5 candidates per step with hidden state continuity

# Accept split parameters from command line or use defaults
SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-10}
GPU_ID=0

mkdir -p results 
# Update reward model for different objectives:
# Helpfulness: Jennny/llama3_8b_helpful_rm_full
# Honesty: Jennny/llama3_8b_honest_rm_full
# Truthfulness: Jennny/llama3_8b_truth_rm_full

CUDA_VISIBLE_DEVICES=${GPU_ID} python mahdpo_hv_orm_guide.py \
    --total_samples 500 \
    --total_splits ${TOTAL_SPLITS} \
    --split_id ${SPLIT_INDEX} \
    --base_model_name meta-llama/Llama-3.1-8B-Instruct \
    --reward_model_name Jennny/llama3_8b_helpful_rm_full \
    --use_base_model_only \
    --num_candidates 5 \
    --dtype bfloat16 \
    --cache_dir /tmp/base_orm_cache_split${SPLIT_INDEX} \
    --output_file results/base_help_orm_guide_split${SPLIT_INDEX}.jsonl \
    --orm_device cuda:0 \
    --max_total_tokens 1024
