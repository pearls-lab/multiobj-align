# Simultaneous Multi-Objective Alignment Across Verifiable and Non-verifiable Rewards

This repository provides data, code, and models for the paper: **Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards** ([arXiv:2510.01167](https://arxiv.org/abs/2510.01167)).

<img src="/main.png" width="600" height="400"/>

## Content

1. [Set Up](#set-up)
2. [Domains and Data](#domains-and-data)
3. [Process Reward Model Training](#process-reward-model-training)
4. [MAH-DPO Training](#mah-dpo-training)
5. [PRM-Guided Decoding](#prm-guided-decoding-with-continuing-hidden-state)
6. [Evaluation](#evaluation)
7. [Trained Model Checkpoints](#trained-model-checkpoints)
8. [Citation](#citation)

## Set Up

This repository is developed under the MosaicML LLM Foundry environment. We recommend using the official Docker image:
```bash
docker pull mosaicml/llm-foundry:2.7.0_cu128-latest
```

Then clone the code:
```bash
git clone https://github.com/pearls-lab/multiobj-align.git
cd multiobj-align
```

Alternatively, install the packages:

```bash
pip install -r requirements.txt
``` 
Our experiments used 8×H100 80GB GPUs.

**Notes:** 
- `mah_dpo/requirements.txt` contains packages specifically for MAH-DPO training

## Domains and Data

Our approach is implemented across three diverse domains, each with multiple objectives:

- **Mathematics**: Mathematical problem-solving with step-by-step reasoning using the [MATH](https://github.com/hendrycks/math) dataset, optimizing for accuracy and engagement.
- **Human Values**: Open-domain conversations using the [UltraFeedback](https://github.com/OpenBMB/UltraFeedback) dataset, optimizing for helpfulness, honesty, and truthfulness.
- **Socratic Mind (AI Tutoring)**: Multi-turn AI tutoring dialogues for teaching programming collected from [Socratic Mind](https://socraticmind.com/), optimizing for accuracy and engagement.


## Process Reward Model Training

We train Process Reward Models (PRMs) across all domains.
```bash
cd prm_train

# Math accuracy (Value model with hindsight relabeling)
bash scripts/train_math_value.sh

# Math engagement (PRM with binary classification)
bash scripts/train_math_engagement.sh

# Human values (Bradley-Terry RM)
bash scripts/train_human_values.sh

# Socratic Mind engagement (PRM with binary classification)
bash scripts/train_sm_engagement.sh

# Socratic Mind accuracy (PRM with binary classification)
bash scripts/train_sm_accuracy.sh

# Unified model (all 7 dimensions)
bash scripts/train_unified_model.sh
```

Note: For Human Values Bradley–Terry RM training, please follow the installation instructions from the [RLHFlow/RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling/tree/main/bradley-terry-rm) repository before running the script.

## MAH-DPO Training

### Download Training Data

The training data for MAH-DPO is available on Google Drive. Download the data and place it in the `mah_dpo/data/` directory:

1. Download the data from [Google Drive](https://drive.google.com/drive/folders/1cpuI6o4NxPy2kSZAT_uDeiJVjTQDL7c8?usp=sharing)
2. Extract the downloaded files
3. Place all files in the `mah_dpo/data/` directory

### MAH-DPO Training

First, perform supervised fine-tuning on base model. Then, train models with objective-specific heads for multi-objective alignment:

```bash
cd mah_dpo
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

# Mathematics (2 Heads)
bash scripts/train_sft_math.sh
bash scripts/train_mahdpo_math.sh

# Human Values (3 Heads) 
bash scripts/train_sft_human_values.sh
bash scripts/train_mahdpo_human_values.sh

# Socratic Mind (2 Heads)
bash scripts/train_sft_sm.sh
bash scripts/train_mahdpo_sm.sh
```

### Inference

Use domain-specific inference scripts multiple configurations (base model, head 0, ensemble, SFT baseline). Edit the script to run the desired configuration or allocate different GPUs for different splits:

```bash
# Mathematics
  bash scripts/inference_math.sh

# Human Values 
  bash scripts/inference_human_values.sh

# Socratic Mind
  ## Step 1: Generate assistant responses (edit scripts/inference_sm.sh to select configuration)
  bash scripts/inference_sm.sh

  ## Step 2: Convert to message format (example with ensemble output)
  python convert_inference_to_messages.py \
    --input results/sm_ensemble_split0.jsonl \
    --output results/sm_ensemble_formatted.jsonl

  ## Step 3: Generate student responses (for evaluation)
  python generate_next_student_response_from_messages.py \
    --input results/sm_ensemble_formatted.jsonl \
    --output results/sm_ensemble_with_student.jsonl
```

## PRM-Guided Decoding with Continuing Hidden State

Our novel test-time optimization method that maintains hidden-state continuity during guided generation.

```bash
cd prm_guided_decoding

# Human Values with ORM Guidance
  ## Base model + ORM guidance
  bash scripts/guide_human_values_base.sh
  ## MAH-DPO model + ORM guidance
  bash scripts/guide_human_values_mahdpo.sh

# Math with Value Model Guidance
  ## Base model + Value model guidance
  bash scripts/guide_math_value_base.sh
  ## MAH-DPO model + Value model guidance
  bash scripts/guide_math_value_mahdpo.sh

# Math with Engagement PRM Guidance
  ## Base model + Engagement PRM guidance
  bash scripts/guide_math_eng_base.sh
  ## MAH-DPO model + Engagement PRM guidance
  bash scripts/guide_math_eng_mahdpo.sh

# Socratic Mind with Candidate Selection
  ## GPT-4o workflow: generate candidates with GPT-4o → select best with PRM → generate student response
  bash scripts/guide_sm.sh
  ## MAH-DPO workflow: generate candidates with base model → select best with PRM → generate student response
  bash scripts/guide_sm_base.sh
  ## MAH-DPO workflow: generate candidates with MAH-DPO → select best with PRM → generate student response
  bash scripts/guide_sm_mahdpo.sh
```

### Unified Model Guidance (All Dimensions)

```bash
# Math domain with unified model
bash scripts/guide_math_unified.sh

# Human values domain with unified model
bash scripts/guide_hv_unified.sh

# Socratic mind domain with candidates generation → select best with unified model → generate student response for later evaluation
bash scripts/guide_sm_unified.sh
```

## Evaluation

All evaluation scripts are in the `eval/` directory.

```bash
cd eval

# Mathematics 
  ## Mathematical Accuracy
    ### Preprocess inference output
    python 01_math500_data_process.py \
      -i results/math_output.jsonl \
      -o results/math_output_processed.json

    ### Evaluate correctness
    python 02_math500_accuracy_eval.py \
      -i results/math_output_processed.json \
      -o results/math_accuracy_eval.json

  ## Mathematical Engagement
  python math500_eng_eval.py \
    results/math_output.jsonl \
    -o results/math_engagement_eval.json

# Human Values
  python hv_eval.py \
    --input_file results/hv_output.jsonl \
    --output_file results/hv_eval.json \
    --model_type dpo \
    --devices cuda:0 cuda:1 cuda:2 \
    --batch_size 4

# Socratic Mind (requires simulated student responses)
  ## Socratic Mind Accuracy
  python reward_model_accuracy_evaluator_stu.py \
    --input results/sm_with_student.jsonl \
    --out results/sm_accuracy_eval.jsonl

  ## Socratic Mind Engagement
  python reward_model_engagement_evaluator_stu.py \
    --input results/sm_with_student.jsonl \
    --out results/sm_engagement_eval.jsonl
```

## Trained Model Checkpoints

We provide trained models for all components. All models are available on HuggingFace Hub and can be loaded directly via `transformers`.

| Mathematics | Human Values | Socratic Mind | Unified |
|-------------|--------------|---------------|---------|
| [SFT](https://huggingface.co/Jennny/dpo_math_new_sft_5e6_2ep) | [SFT](https://huggingface.co/Jennny/clean_len_human_value_sft_llama8b_5e7_1ep_192_step-4992) | [SFT](https://huggingface.co/Jennny/sm_sft_qwen25_fsdp_5e6_2ep_bz128_LATEST_updated) | [Unified PRM](https://huggingface.co/andre930/my_unified_model_classification_all_2108) |
| [MAH-DPO](https://huggingface.co/Jennny/good_sft_mdpo_step-LATEST) | [MAH-DPO](https://huggingface.co/Jennny/new_sft_mdpo_llama8b_3heads_help_hon_truth5e7_1ep_bz120_step19920) | [MAH-DPO](https://huggingface.co/Jennny/sm_mdpo_qwen25_fsdp_5e6_2ep_bz128_LATEST_with_SFT_updated) | |
| [Accuracy PRM](https://huggingface.co/Qwen/Qwen2.5-Math-PRM-7B) | [Helpfulness PRM](https://huggingface.co/Jennny/llama3_8b_helpful_rm_full) | [Engagement PRM](https://huggingface.co/Jennny/eng_rm_1e5_700) | |
| [Accuracy Value Model](https://huggingface.co/Jennny/qwen-math-value-model-join-09-5e5-3ep) | [Honesty PRM](https://huggingface.co/Jennny/llama3_8b_honest_rm_full) | [Accuracy PRM](https://huggingface.co/andre930/acc_latest_filtered) | |
| [Engagement PRM](https://huggingface.co/andre930/eng_prm) | [Truthfulness PRM](https://huggingface.co/Jennny/llama3_8b_truth_rm_full) | | |

## Citation

If you found our work useful please consider citing it:

```bibtex
@article{shen2025simultaneous,
  title={Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards},
  author={Shen, Yiran and Xia, Yu and Chang, Jonathan and Ammanabrolu, Prithviraj},
  journal={arXiv preprint arXiv:2510.01167},
  year={2025},
  url={https://arxiv.org/abs/2510.01167}
}
```
