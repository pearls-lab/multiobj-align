#!/bin/bash
# Multi-Action-Head DPO Training for Human Values
# 3 heads: Helpfulness, Honesty, Truthfulness
# Requires: SFT checkpoint (update model.archive path)

# Update this path to your SFT checkpoint
SFT_CHECKPOINT="Jennny/clean_len_human_value_sft_llama8b_5e7_1ep_192_step-4992"

python -u mahdpo_train.py \
  model=llama8b \
  +model.num_heads=3 \
  datasets=[data/ultra_help.json,data/ultra_hon.json,data/ultra_truth.json] \
  loss=dpo \
  loss.beta=0.1 \
  exp_name=mahdpo_llama8b_3heads_help_hon_truth \
  lr=5e-7 \
  gradient_accumulation_steps=4 \
  batch_size=30 \
  eval_batch_size=15 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  n_epochs=1 \
  max_prompt_length=256 \
  max_length=768 \
  model.archive=${SFT_CHECKPOINT}
