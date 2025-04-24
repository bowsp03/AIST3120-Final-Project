# AIST3120-Final-Project
This repository is used for AIST3120 final project

# Noisy-Label Learning for NER with LoRA Adaptation

This repository implements a noise-robust Named Entity Recognition (NER) system using BERT with LoRA (Low-Rank Adaptation), based on the paper ["Learning from Noisy Labels for Entity-Centric Information Extraction"](https://arxiv.org/abs/2104.08656).

## Key Features
- 🛡️ Robust training against label noise using co-regularization
- 🚀 Efficient fine-tuning with LoRA (reduces trainable parameters by ~90%)
- 📊 Integrated resource tracking (GPU/CPU memory, time)
- 🔧 Flexible configuration for different BERT variants
- 📈 WandB integration for experiment tracking

## Requirements
- Python 3.8+
- PyTorch 1.12+
- Transformers 4.28+
- PEFT (for LoRA)
- seqeval
- wandb

Install dependencies:
```bash
pip install torch transformers peft seqeval wandb truecase psutil

Dataset Preparation

Place your CoNLL-format files in ./data/ with these names:

train.txt - Training data
dev.txt - Validation data
test.txt - Test data
conllpp_test.txt - Corrected test set (CrossWeigh)

python train.py \
  --model_name_or_path bert-base-cased \
  --batch_size 64 \
  --learning_rate 1e-5 \
  --num_train_epochs 500 \
  --n_model 2 \
  --alpha 50.0

Project Root/
│
├── LoRA/                       # LoRA-related files
│   ├── __pycache__/            # Cache folder for Python bytecode
│   ├── data/                   # Folder for storing data
│   ├── model.py                # Python script for model definition
│   ├── prepro.py               # Python script for preprocessing
│   ├── train-lr-Copy1.py       # Python script (likely a copy of train-lr.py)
│   ├── train-lr.py             # Python script for training with learning rate
│   ├── train.py                # Python script for training
│   ├── utils.py                # Python script for utility functions
│   └── Readme.txt              # README file with project details
│
└── Scheduler/                  # Scheduler-related files
    ├── __pycache__/            # (Assumed) Cache folder for Python bytecode
    ├── scheduler.py            # (Assumed) Main script for scheduling training jobs
    ├── config/                 # (Assumed) Folder for configuration files
    │   ├── scheduler_config.yaml  # (Assumed) YAML file for scheduler settings
    │   └── training_params.yaml   # (Assumed) YAML file for training parameters
    ├── logs/                   # (Assumed) Folder for scheduler logs
    │   ├── scheduler.log       # (Assumed) Log file for scheduler activity
    │   └── error.log           # (Assumed) Log file for errors
    ├── tasks.py                # (Assumed) Script defining scheduling tasks
    └── utils.py                # (Assumed) Utility functions for the scheduler
