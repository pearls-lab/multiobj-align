#!/bin/bash
# Train Human Values Reward Models (Bradley-Terry)
# Three separate models: Helpfulness, Honesty, Truthfulness

# Train Helpfulness RM
python hv_rm.py \
    --max_length 2048 \
    --train_set_path Jennny/ultrafeedback_binarized_helpfulness_prefs \
    --eval_set_path Jennny/ultrafeedback_binarized_helpfulness_prefs \
    --deepspeed deepspeed_config.json \
    --output_path ./models/llama3_8b_helpful_rm \
    --hub_repo_name llama3_8b_helpful_rm \
    --wandb_project hv_rm

# Train Honesty RM
python hv_rm.py \
    --max_length 2048 \
    --train_set_path Jennny/ultrafeedback_binarized_honesty_prefs \
    --eval_set_path Jennny/ultrafeedback_binarized_honesty_prefs \
    --deepspeed deepspeed_config.json \
    --output_path ./models/llama3_8b_honest_rm \
    --hub_repo_name llama3_8b_honest_rm \
    --wandb_project hv_rm

# Train Truthfulness RM
python hv_rm.py \
    --max_length 2048 \
    --train_set_path Jennny/ultrafeedback_binarized_truthfulness_prefs \
    --eval_set_path Jennny/ultrafeedback_binarized_truthfulness_prefs \
    --deepspeed deepspeed_config.json \
    --output_path ./models/llama3_8b_truth_rm \
    --hub_repo_name llama3_8b_truth_rm \
    --wandb_project hv_rm
