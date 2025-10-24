#!/bin/bash
# Train Math Value Model with Hindsight Relabeling
# Uses Qwen2.5-Math-PRM-7B backbone with trainable value head
# Discount factor gamma=0.9

python math_acc_value.py \
  --model_name "Qwen/Qwen2.5-Math-PRM-7B" \
  --data_files mc_data_09.json \
  --output_dir "./models/math_value_model" \
  --experiment_name "qwen-math-value-09" \
  --batch_size 16 \
  --learning_rate 5e-5 \
  --num_epochs 3 \
  --gradient_accumulation_steps 2 \
  --push_to_hub \
  --hub_model_id "your_username/qwen-math-value-model-09"
