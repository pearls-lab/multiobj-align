#!/bin/bash
# Complete Socratic Mind Workflow with MAH-DPO Model + Unified Model Selection
# 1. Generate 5 candidates using MAH-DPO ensemble model
# 2. Select best candidate using unified model (all 7 dimensions)
# 3. Generate next student response

# Accept split parameters from command line or use defaults
SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-5}

# Update these paths
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DPO_CHECKPOINT="Jennny/sm_mdpo_qwen25_fsdp_5e6_2ep_bz128_LATEST_with_SFT_updated"
DATA_FILE="sm_acc.json"
CANDIDATES_FILE="results/sm_mahdpo_unified_candidates_split${SPLIT_INDEX}.jsonl"
TRANSFORMED_CANDIDATES_FILE="results/sm_mahdpo_unified_candidates_transformed_split${SPLIT_INDEX}.jsonl"
BEST_FILE="results/sm_mahdpo_unified_best_split${SPLIT_INDEX}.jsonl"
FINAL_FILE="results/sm_mahdpo_unified_with_student_split${SPLIT_INDEX}.jsonl"
GPU_ID=0

mkdir -p results 

# Unified model for holistic quality assessment
UNIFIED_MODEL="andre930/my_unified_model_classification_all_2108"

SYSTEM_PROMPT="You are a tutor who is helping a beginner student learn programming. Continue as the same tutor and reply similarly to the last student message, matching EXACTLY the SAME speaking tone and tutoring style as in your earlier messages (e.g. reply to the student's last message concisely in 1-2 sentences and then always ask a meaningful follow-up question)."

echo "Step 1: Generating 5 candidate responses with MAH-DPO ensemble..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python mahdpo_sm_prm_guide.py \
  --model_type dpo \
  --model_name ${BASE_MODEL} \
  --checkpoint_path ${DPO_CHECKPOINT} \
  --num_heads 2 \
  --use_ensemble \
  --data_file ${DATA_FILE} \
  --num_samples 100 \
  --num_candidates 5 \
  --total_splits ${TOTAL_SPLITS} \
  --split_index ${SPLIT_INDEX} \
  --max_new_tokens 300 \
  --temperature 1.0 \
  --top_p 1.0 \
  --top_k 50 \
  --dtype bfloat16 \
  --cache_dir /tmp/model_cache_split${SPLIT_INDEX} \
  --output_file ${CANDIDATES_FILE} \
  --system_prompt "${SYSTEM_PROMPT}"

echo "Step 1.5: Transforming candidates for unified model scoring..."
python transform.py ${CANDIDATES_FILE} ${TRANSFORMED_CANDIDATES_FILE}

echo "Step 2: Selecting best candidate with unified model (${UNIFIED_MODEL})..."
python sm_score_candidate_unified.py \
  --input ${TRANSFORMED_CANDIDATES_FILE} \
  --output ${BEST_FILE} \
  --model ${UNIFIED_MODEL} \
  --max_length 2048 \
  --batch_size 8 \
  --num_gpus 8 \
  --strategy multi_process

echo "Step 3: Generating next student response..."
python generate_next_student_response_gpt4o.py \
  --input ${BEST_FILE} \
  --output ${FINAL_FILE} \
  --model gpt-4o

echo "Complete! Output saved to ${FINAL_FILE}"

