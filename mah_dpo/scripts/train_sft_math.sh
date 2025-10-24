#!/bin/bash
# Supervised Fine-Tuning for Math and Socratic Mind (Qwen2.5-7B)
# This script fine-tunes Qwen2.5-7B on math and tutoring datasets

python -u train.py \
  model=qwen7b \
  datasets=[data/math_combined_clean.json] \
  loss=sft \
  exp_name=qwen25_math_sft \
  gradient_accumulation_steps=4 \
  batch_size=64 \
  eval_batch_size=32 \
  trainer=FSDPTrainer \
  sample_during_eval=false \
  eval_every=1_000 \
  n_epochs=2 \
  lr=5e-6 \
  max_prompt_length=512 \
  max_length=2560
