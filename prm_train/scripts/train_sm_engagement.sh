#!/bin/bash
# Train Socratic Mind Engagement PRM
# Binary classification on multi-turn tutoring dialogues
# Dataset: Jennny/final_eng_rate_latest2

python sm_eng_prm.py

# Note: Configuration is in the main() function of sm_eng_prm.py
# Default settings:
#   - Learning rate: 1e-5
#   - Batch size: 32
#   - Epochs: 5
#   - Dataset: Jennny/final_eng_rate_latest2
