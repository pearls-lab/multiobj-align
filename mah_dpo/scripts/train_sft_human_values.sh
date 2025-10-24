#!/bin/bash
# Supervised Fine-Tuning for Human Values (Llama-3.1-8B)
# This script fine-tunes Llama-3.1-8B on UltraFeedback chosen responses

python -u train.py \
  model=llama8b \
  datasets=[data/ultra_combined_clean.json] \
  loss=sft \
  exp_name=llama8b_ultrafb_sft \
  gradient_accumulation_steps=4 \
  batch_size=32 \
  eval_batch_size=16 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  eval_every=5_000 \
  n_epochs=1 \
  lr=5e-7 \
  max_prompt_length=256 \
  max_length=768 
