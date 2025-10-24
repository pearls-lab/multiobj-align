#!/bin/bash
# Train Math Engagement PRM
# Binary classification on incremental step sequences
# Base model: Llama-3.1-8B

# Step 1: Process data
python unified_data_processor.py \
  --prm_sample_size 99999 \
  --value_sample_size 0 \
  --preference_sample_size 0 \
  --engagement_sample_size 0 \
  --accuracy_sample_size 0 \
  --output_dir ./models/math_engagement_data

# Step 2: Train model
python unified_reward_model.py \
  --processed_data_dir ./models/math_engagement_data \
  --output_dir ./models/math_engagement_model \
  --num_epochs 2 \
  --batch_size 128 \
  --learning_rate 1e-5
