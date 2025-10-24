#!/bin/bash
# Complete Socratic Mind Workflow with GPT-4o + Candidate Selection
# 1. Generate 5 candidates using GPT-4o
# 2. Select best candidate using engagement or accuracy RM
# 3. Generate next student response

# Update these paths
INPUT_FILE="preference_prefixes_500.jsonl"
CANDIDATES_FILE="preference_prefixes_500_candidates.jsonl"
BEST_FILE="results/preference_prefixes_500_best.jsonl"
FINAL_FILE="results/preference_prefixes_500_with_student.jsonl"

mkdir -p results 

# Choose PRM model: Engagement or Accuracy
PRM_MODEL="Jennny/eng_rm_1e5_700"  # or "andre930/acc_latest_filtered"

# OpenAI API key for GPT-4o (set here or override via environment)
export OPENAI_API_KEY=${OPENAI_API_KEY:-"YOUR_OPENAI_API_KEY"}

echo "Step 1: Generating 5 candidate responses with GPT-4o..."
python generate_5_candidates_gpt4o.py \
  --input ${INPUT_FILE} \
  --output ${CANDIDATES_FILE} \
  --model gpt-4o \
  --num_candidates 5

echo "Step 2: Selecting best candidate with PRM (${PRM_MODEL})..."
python sm_score_candidate.py \
  --input ${CANDIDATES_FILE} \
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
