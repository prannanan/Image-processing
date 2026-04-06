# House Recognition — Binary Image Classification

**SuperAI Engineer Season 6 | Individual Hackathon**

Binary classification task: predict whether an image is a **house (1)** or **not a house (0)**.  
Evaluated using **Accuracy Score**.

---

## Project Structure

```
Image processing/
├── House Recognition.ipynb          # Main notebook (training + inference)
├── instructions.md                  # Competition description
└── super-ai-engineer-season-6-individual-hackathon-house-recognition/
    ├── train.csv                    # Training labels (image_name, class)
    ├── sample_submission.csv        # Submission format template
    ├── submission.csv               # Final predictions
    ├── best_model.pth               # Saved best model checkpoint
    ├── train/train/                 # Training images
    └── test/test/                   # Test images
```

---

## Approach

| Component | Choice |
|-----------|--------|
| Architecture | EfficientNet-B0 (pretrained on ImageNet) |
| Classifier head | `Dropout(0.3) → Linear(1280 → 1)` |
| Loss function | `BCEWithLogitsLoss` |
| Optimizer | AdamW (`lr=1e-4`, `weight_decay=1e-4`) |
| LR Schedule | CosineAnnealingLR (T_max = 15) |
| Metric | Accuracy |

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| `IMG_SIZE` | 224 |
| `BATCH_SIZE` | 32 |
| `EPOCHS` | 15 |
| `LR` | 1e-4 |
| `VAL_SPLIT` | 20% (stratified) |
| `SEED` | 42 |

---

## Data Pipeline

**Training augmentations:**
- Random horizontal flip
- Random vertical flip (p=0.1)
- Color jitter (brightness, contrast, saturation ±0.2)
- Random rotation (±15°)
- Normalize with ImageNet mean/std

**Validation / Test transforms:**
- Resize to 224×224
- Normalize with ImageNet mean/std

---

## Training

The model is trained for 15 epochs with best-model checkpointing:
- After each epoch, validation accuracy is evaluated
- If it improves, weights are saved to `best_model.pth`
- Training/validation loss and accuracy are logged per epoch

---

## Inference & Submission

1. Load `best_model.pth`
2. Run inference on all test images
3. Apply sigmoid → threshold at 0.5 → binary label
4. Align results with `sample_submission.csv` order
5. Save as `submission.csv` with columns `id` and `answer`

---

## Requirements

```
torch
torchvision
numpy
pandas
Pillow
scikit-learn
matplotlib
```

Install with:
```bash
pip install torch torchvision numpy pandas Pillow scikit-learn matplotlib
```

---

## Usage

Open and run all cells in **`House Recognition.ipynb`** sequentially.  
A GPU (CUDA) is recommended but CPU is supported automatically.
