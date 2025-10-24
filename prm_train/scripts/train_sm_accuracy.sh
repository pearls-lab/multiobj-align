#!/bin/bash
# Train Socratic Mind Accuracy PRM
# Binary classification on multi-turn tutoring dialogues (multi-turn only)
# Dataset: Jennny/final_acc_rate_latest3

python sm_acc_prm.py

# Note: Configuration is in the main() function of sm_acc_prm.py
# Default settings:
#   - Learning rate: 5e-6
#   - Batch size: 32
#   - Epochs: 3
#   - Dataset: Jennny/final_acc_rate_latest3
#   - Special: Filters samples with ≤1 Assistant turn
