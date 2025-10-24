#!/bin/bash
# Multi-Action-Head DPO Training for Socratic Mind (AI Tutoring)
# 2 heads: Accuracy, Engagement
# Requires: SFT checkpoint (update model.archive path)

# Update this path to your SFT checkpoint
SFT_CHECKPOINT="Jennny/sm_sft_qwen25_fsdp_5e6_2ep_bz128_LATEST_updated"

python -u mahdpo_train.py \
  model=qwen7b \
  +model.num_heads=2 \
  datasets=[data/sm_acc.json,data/sm_eng.json] \
  loss=dpo \
  loss.beta=0.1 \
  exp_name=mahdpo_qwen25_2heads_sm \
  lr=5e-6 \
  gradient_accumulation_steps=4 \
  batch_size=64 \
  eval_batch_size=32 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  model.fsdp_policy_mp=bfloat16 \
  n_epochs=2 \
  max_prompt_length=1336 \
  max_length=1536 \
  eval_every=200 \
  model.archive=${SFT_CHECKPOINT}

