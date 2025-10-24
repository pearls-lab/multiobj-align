#!/bin/bash
# Complete Socratic Mind Workflow with Base Model + Candidate Selection
# 1. Generate 5 candidates using base model
# 2. Select best candidate using engagement or accuracy RM
# 3. Generate next student response

# Accept split parameters from command line or use defaults
SPLIT_INDEX=${1:-0}
TOTAL_SPLITS=${2:-5}

mkdir -p results 

# Update these paths
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
DATA_FILE="sm_acc.json"
CANDIDATES_FILE="results/sm_base_candidates_split${SPLIT_INDEX}.jsonl"
TRANSFORMED_CANDIDATES_FILE="results/sm_base_candidates_transformed_split${SPLIT_INDEX}.jsonl"
BEST_FILE="results/sm_base_best_split${SPLIT_INDEX}.jsonl"
FINAL_FILE="results/sm_base_with_student_split${SPLIT_INDEX}.jsonl"
GPU_ID=0

# Choose PRM model: Engagement or Accuracy
PRM_MODEL="Jennny/eng_rm_1e5_700"  # or "andre930/acc_latest_filtered"

SYSTEM_PROMPT="You are a tutor who is helping a beginner student learn programming. Continue as the same tutor and reply similarly to the last student message, matching EXACTLY the SAME speaking tone and tutoring style as in your earlier messages (e.g. reply to the student's last message concisely in 1-2 sentences and then always ask a meaningful follow-up question)."

echo "Step 1: Generating 5 candidate responses with base model..."
CUDA_VISIBLE_DEVICES=${GPU_ID} python mahdpo_sm_prm_guide.py \
  --model_type base \
  --model_name ${BASE_MODEL} \
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

echo "Step 1.5: Transforming candidates for PRM scoring..."
python transform.py ${CANDIDATES_FILE} ${TRANSFORMED_CANDIDATES_FILE}

echo "Step 2: Selecting best candidate with PRM (${PRM_MODEL})..."
python sm_score_candidate.py \
  --input ${TRANSFORMED_CANDIDATES_FILE} \
  --output ${BEST_FILE} \
  --model ${PRM_MODEL} \
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


