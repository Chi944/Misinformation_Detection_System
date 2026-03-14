# Kaggle BERT retraining: class imbalance fix

If BERT collapses to predicting one class (e.g. all P(misinfo) ≈ 0), the usual cause is **class imbalance** in the Kaggle training set. Fix by using **class weights** or **rebalanced data**.

## 1. Diagnose imbalance locally

From the repo root, run the balance script on your Kaggle train CSV (or copy it into `data/`):

```bash
python scripts/check_label_balance.py /path/to/kaggle_train.csv
```

Example output:

```
kaggle_train.csv: total=50000  0 (credible)=45000 (90.0%)  1 (misinfo)=5000 (10.0%)  ratio 1:0=0.11
  -> WARNING: severe imbalance; use class weights or oversampling when training BERT.
```

If one class is &gt; ~70%, use the steps below.

## 2. Class weights (recommended)

Use inverse-frequency weights so the loss pays more attention to the minority class.

**Formula:**  
`weight[class] = total_samples / (num_classes * count_class)`  
For binary: `w0 = n / (2 * n0)`, `w1 = n / (2 * n1)`.

**In your Kaggle notebook (PyTorch):**

```python
import torch
import torch.nn.functional as F

# After building train_loader or having train_labels list:
train_labels = ...  # list/int array of 0 and 1
n0 = (np.array(train_labels) == 0).sum()
n1 = (np.array(train_labels) == 1).sum()
n = len(train_labels)
w0 = n / (2.0 * max(1, n0))
w1 = n / (2.0 * max(1, n1))
class_weights = torch.tensor([w0, w1], dtype=torch.float32).to(device)

# In the training loop, use:
loss = F.cross_entropy(logits, labels, weight=class_weights)
```

**In this repo (local/Kaggle):**

- **`train_all` (MasterTrainer):** Class weights are computed automatically from `train_labels` and passed to `F.cross_entropy` when training BERT. No extra args needed.
- **`BERTTrainer` (src/models/bert_classifier.py):** You can pass `class_weights` to the constructor and the trainer will use them in its loss.

## 3. Rebalanced data (optional)

Alternatively (or in addition), oversample the minority class so each batch has a more even split:

- **Option A:** Duplicate minority-class rows in the training DataFrame until the ratio is closer to 50/50, then build the DataLoader as usual.
- **Option B:** Use a **WeightedRandomSampler** so that each batch is more likely to include minority samples:

```python
from torch.utils.data import WeightedRandomSampler

# For each sample index, assign weight = 1/count_of_that_class
class_counts = np.bincount(train_labels)
weights = 1.0 / np.array([class_counts[l] for l in train_labels])
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
train_loader = DataLoader(dataset, batch_size=16, sampler=sampler)
```

## 4. Label convention

This codebase uses:

- **0 = credible**, **1 = misinformation**
- BERT output index **1** = P(misinformation). The detector uses `probs[0, 1]` as the misinfo confidence.

Ensure your Kaggle dataset and loss use the same convention (0/1) and that you do not flip labels when saving the checkpoint.

## 5. After retraining on Kaggle

1. Download the new `bert_classifier.pt` and place it in `models/`.
2. Run a quick check:

   ```bash
   python scripts/check_label_balance.py data/train.csv data/val.csv data/test.csv
   ```

3. Verify BERT is no longer collapsed (e.g. run the 4-example BERT confidence script from the project root; credible examples should get lower P(misinfo), misinfo examples higher).
