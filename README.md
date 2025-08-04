# EfficientNet Fine-Tuning (PyTorch) — Stanford Dogs

## Overview

This repository is a **complete PyTorch implementation** of image classification via transfer learning with EfficientNet (originally from a Keras example). It includes advanced regularization and optimization techniques:
- MixUp / CutMix
- Exponential Moving Average (EMA) of weights
- OneCycleLR learning rate schedule
- Layer-wise learning rates
- Gradient accumulation
- Hyperparameter sweep (Optuna)
- AMP (mixed precision)
- Label smoothing
- Structured logging and checkpointing

## Project Structure
```
Efficientnet-Finetune/
├── main.py                   # Training entry point
├── sweep.py                 # Optuna hyperparameter sweep
├── prepare_data.py          # Downloads & organizes Stanford Dogs for PyTorch
├── predict.py               # Inference on a single image
├── config.yaml              # Default hyperparameters
├── requirements.txt         # Python dependencies
├── README.md
├── utils/                   # Helpers (logging, augmentations, EMA)
├── data/                    # Dataset loader
├── models/                  # EfficientNet classifier
├── trainer/                 # Training loop
├── checkpoints/             # Saved model checkpoints
└── sweep_runs/             # Sweep outputs
```
## Feature Spotlight

### 🎨 Smart Augmentation: MixUp & CutMix
Blend samples and their labels to fabricate richer training signal. These augmentations act like creativity boosts—softening decision boundaries, reducing overfitting, and making the model less brittle on small data.

### 🧊 Weight Smoothing: EMA (Exponential Moving Average)
Maintain a trailing, smoothed copy of model parameters during training. At evaluation time the EMA snapshot is used to give more stable and generally better validation performance than raw weights.

### ⚡ Learning Rate Magic: OneCycleLR + Layer-wise Rates
Training uses a dynamic OneCycleLR schedule that warms up then cools down for sharper convergence. Backbone and head get different learning rates (smaller for the pretrained features, larger for the newly initialized classifier) so you tune only what needs tuning without destabilizing powerful priors.

### 🛠️ Regularization Arsenal
- **Label Smoothing:** Prevents overconfident predictions by softening hard targets.  
- **Stochastic Depth (`drop_path_rate`):** Randomly drops residual paths in EfficientNet during training to improve robustness.  
- **Dropout:** Applied on the custom head to further guard against overfitting.  

### 🧮 Effective Batch Scaling: Gradient Accumulation
Simulate large batches while keeping per-step memory low. Gradients are accumulated across multiple mini-batches before a weight update, letting you train with “big-batch” behavior even on constrained hardware.

### 🧪 Precision & Performance: AMP (Automatic Mixed Precision)
Automatically uses lower-precision arithmetic when beneficial—shrinking memory usage and speeding up computation without manual intervention when CUDA is available.

### 🔍 Automated Tuning: Optuna Sweep
Hyperparameters like base learning rate, backbone multiplier, dropout, stochastic depth, MixUp/CutMix strength, label smoothing, and weight decay are explored systematically to find strong settings with minimal manual searching.

### 🧩 Optimized Optimization: AdamW + Parameter Grouping
Uses AdamW for proper decoupled weight decay. Parameter groups are carefully constructed so that norm/bias terms are excluded from decay and backbone vs head get their own learning rate schedules.

### 🏁 Resilience & Observability
- **Checkpointing:** Best model (by validation accuracy) is saved, along with full training state.  
- **Logging:** Structured logs capture epoch-level metrics, hyperparameters, and decisions.  
- **Early Stopping (configurable):** Avoids wasted epochs when validation plateaus.

## Installation

```bash
git clone <your-repo-url>
cd efficientnet_finetune
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data Preparation
```
python prepare_data.py --output data/stanford_dogs_prepared --img-size 224

This uses tensorflow_datasets to download stanford_dogs and saves images into:

    data/stanford_dogs_prepared/
      train/<breed>/*.jpg
      val/<breed>/*.jpg
```

## Training
```
Edit or rely on config.yaml. Example:
python main.py --config config.yaml
Or override fields by editing config.yaml. Key fields include:
    •	lr, backbone_lr_mult, mixup_alpha, cutmix_alpha
    •	epochs, batch_size, label_smoothing
    •	use_amp, ema_decay

Logs and checkpoints go to the path under output_dir (default: checkpoints/run_default).
```

## Hyperparameter Sweep
```
Run automated tuning:
    python sweep.py
```

## Inference
```
Load a trained checkpoint and predict:
python predict.py --image path/to/dog.jpg --checkpoint checkpoints/run_default/best.pth --img-size 224
```



