#!/bin/bash
# Math Engagement: MAH-DPO Model + Engagement PRM Guidance

# Accept split parameters from command line or use defaults
SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-10}

# Update these paths
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DPO_CHECKPOINT="Jennny/good_sft_mdpo_step-LATEST"
ENG_PRM="andre930/eng_prm"
BASE_GPU_ID=0
REWARD_GPU_ID=1

mkdir -p results 

CUDA_VISIBLE_DEVICES=${BASE_GPU_ID},${REWARD_GPU_ID} python mahdpo_math_eng_guide.py \
  --base_model_name ${BASE_MODEL} \
  --use_mdpo_model \
  --dpo_checkpoint_path ${DPO_CHECKPOINT} \
  --mdpo_head_mode ensemble \
  --num_heads 2 \
  --unified_reward_model_name ${ENG_PRM} \
  --use_reward_guidance \
  --candidate_selection unified_guided \
  --num_samples 500 \
  --split_index ${SPLIT_INDEX} \
  --total_splits ${TOTAL_SPLITS} \
  --output_file results/mdpo_eng_prm_guide_split${SPLIT_INDEX}.jsonl \
  --base_device_id 0 \
  --reward_device_id 1 \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --seed 42
