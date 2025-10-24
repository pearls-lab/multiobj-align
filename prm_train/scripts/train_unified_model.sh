#!/bin/bash
# Train Unified Reward Model (All 7 Dimensions)
# Covers: math engagement, math accuracy, helpfulness, honesty, 
#         truthfulness, SM engagement, SM accuracy
# Total: 168,514 examples

# Step 1: Process data from all sources
python unified_data_processor.py \
  --prm_sample_size 99999 \
  --preference_sample_size 99999 \
  --value_sample_size 99999 \
  --engagement_sample_size 99999 \
  --accuracy_sample_size 99999 \
  --output_dir ./unified_data

# Step 2: Train unified model
python unified_reward_model.py \
  --processed_data_dir ./unified_data \
  --output_dir ./models/unified_model \
  --num_epochs 2 \
  --batch_size 128 \
  --learning_rate 1e-5

