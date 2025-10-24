#!/bin/bash
# Multi-Action-Head DPO Training for Mathematical Reasoning
# 2 heads: Accuracy, Engagement
# Requires: SFT checkpoint (update model.archive path)

# Update this path to your SFT checkpoint
SFT_CHECKPOINT="Jennny/dpo_math_new_sft_5e6_2ep"

python -u mahdpo_train.py \
  model=qwen7b \
  +model.num_heads=2 \
  datasets=[data/math_acc.json,data/math_eng.json] \
  loss=dpo \
  loss.beta=0.1 \
  exp_name=mahdpo_qwen25_2heads_math \
  lr=1e-6 \
  gradient_accumulation_steps=4 \
  batch_size=32 \
  eval_batch_size=16 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  n_epochs=2 \
  max_prompt_length=512 \
  max_length=1536 \
  model.archive=${SFT_CHECKPOINT}