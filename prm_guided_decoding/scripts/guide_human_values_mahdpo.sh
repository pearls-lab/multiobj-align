#!/bin/bash
# PRM-Guided Decoding for Human Values (MAH-DPO Model + ORM)
# Uses MAH-DPO checkpoint with ensemble + Helpfulness ORM guidance
# 5 candidates per step with hidden state continuity

# Accept split parameters from command line or use defaults
SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-10}
GPU_ID=0

mkdir -p results 
# Update checkpoint path to your trained MAH-DPO model, need to change the perturbation scale in multihead_model to 0.005
CHECKPOINT_PATH="Jennny/new_sft_mdpo_llama8b_3heads_help_hon_truth5e7_1ep_bz120_step19920"

CUDA_VISIBLE_DEVICES=${GPU_ID} python mahdpo_hv_orm_guide.py \
    --total_samples 500 \
    --total_splits ${TOTAL_SPLITS} \
    --split_id ${SPLIT_INDEX} \
    --model_name ${CHECKPOINT_PATH} \
    --base_model_name meta-llama/Llama-3.1-8B-Instruct \
    --reward_model_name Jennny/llama3_8b_helpful_rm_full \
    --num_heads 3 \
    --use_ensemble \
    --num_candidates 5 \
    --dtype bfloat16 \
    --cache_dir /tmp/mdpo_orm_cache_split${SPLIT_INDEX} \
    --output_file results/mdpo_help_orm_guide_split${SPLIT_INDEX}.jsonl \
    --orm_device cuda:0 \
    --max_total_tokens 1024
