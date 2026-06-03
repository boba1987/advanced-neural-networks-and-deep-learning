# Customer Support Ticket Classifier

Classify support tickets into one of four types — **Incident**, **Request**, **Problem**, or **Change** — using only the ticket **body** text.

Built as a hands-on deep learning project: a classic **TF-IDF + PyTorch MLP** pipeline (see [`Customer_Support_Ticket_Classifier_TF_IDF.ipynb`](Customer_Support_Ticket_Classifier_TF_IDF.ipynb)) and **BERT** fine-tuning notebook ([`Customer_Support_Ticket_Classifier_BERT.ipynb`](Customer_Support_Ticket_Classifier_BERT.ipynb)), trained on English support tickets and runnable in Google Colab.

**Test set results (2,385 tickets):** MLP **82.76%** macro F1 · BERT **83.67%** macro F1 (+0.91 pp)

| Notebook | Colab |
|----------|-------|
| TF-IDF + MLP | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boba1987/advanced-neural-networks-and-deep-learning/blob/main/Customer_Support_Ticket_Classifier_TF_IDF.ipynb) |
| BERT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boba1987/advanced-neural-networks-and-deep-learning/blob/main/Customer_Support_Ticket_Classifier_BERT.ipynb) |

## What this project does

This project trains a model to predict the type from the ticket message alone.

**Input:** ticket body (`body` column)  
**Output:** one of `Incident`, `Request`, `Problem`, `Change`

Both notebooks have the same setup:

1. Load and explore the dataset  
2. Clean text, encode labels, split train / validation / test (80% / 10% / 20%, stratified)  

**[`Customer_Support_Ticket_Classifier_TF_IDF.ipynb`](Customer_Support_Ticket_Classifier_TF_IDF.ipynb)** — TF-IDF + PyTorch MLP:

3. Build vectorizer and MLP using the **best config from hyperparameter search** (`max_features_15k`; six candidates listed for reference)  
4. Train with early stopping (validation **macro F1**)  
5. Evaluate on the held-out test set  
6. Save the model and run sample predictions  

**[`Customer_Support_Ticket_Classifier_BERT.ipynb`](Customer_Support_Ticket_Classifier_BERT.ipynb)** — BERT fine-tuning:

3. Fine-tune `bert-base-uncased` on the same split (GPU recommended)  
4. Evaluate on the held-out test set  
5. Save the model and run sample predictions  

## Dataset

| File | Rows | Columns |
|------|------|---------|
| `dataset-tickets-en.csv` | 11,921 | `body`, `type` |

**Class distribution:**

| Type | Count |
|------|------:|
| Incident | 4,642 |
| Request | 3,498 |
| Problem | 2,497 |
| Change | 1,284 |

`Change` is the smallest class, so **macro F1** (average F1 across classes) is a better tuning metric than accuracy alone.

## What we were trying to do

The project has two parts on the **same dataset and split**:

**1. Baseline without transformers** ([`Customer_Support_Ticket_Classifier_TF_IDF.ipynb`](Customer_Support_Ticket_Classifier_TF_IDF.ipynb))

- Build a strong, lightweight classifier: **TF-IDF bag-of-ngrams → PyTorch MLP**
- Tune it properly — compare six configs in a structured search and pick the best by **validation macro F1**
- Answer: *how much can a simple model achieve?* → **82.76%** test macro F1 after tuning and full training

**2. Transformer comparison** ([`Customer_Support_Ticket_Classifier_BERT.ipynb`](Customer_Support_Ticket_Classifier_BERT.ipynb))

- Fine-tune **`bert-base-uncased`** on the identical split
- Compare against the tuned MLP — on test, BERT reaches **83.67%** macro F1 vs the MLP’s **82.76%** (+0.91 pp)

## Experiment: hyperparameter search

Six configs were compared on the same train/validation split. Search used up to **12 epochs** per config; the winner was retrained in Phase 4 for up to **25 epochs**.

| Config | Val accuracy | Val macro F1 | TF-IDF features |
|--------|-------------:|-------------:|----------------:|
| **max_features_15k** ✓ | 82.29% | **82.60%** | 15,000 |
| wider_net_small_batch | **82.70%** | 82.23% | 10,000 |
| wider_net_lower_lr | 82.49% | 82.20% | 10,000 |
| more_features | 81.76% | 81.94% | 10,000 |
| longer_ngrams | 80.29% | 79.80% | 8,000 |
| baseline | 78.09% | 79.36% | 5,000 |

**Selected config:** `max_features_15k`

```
max_features = 15000
ngram_range  = (1, 3)
hidden       = (512, 256, 128)
dropout      = (0.3, 0.2, 0.1)
lr           = 5e-4
batch_size   = 64
```

### Findings

1. **Tuning helped.** The baseline (5k features, smaller network) reached **79.36%** validation macro F1. The best config reached **82.60%** — about **+3.2 percentage points** from hyperparameter search alone.

2. **More TF-IDF features matter.** Moving from 5k → 10k → 15k features consistently improved scores. Ticket types are often distinguished by specific phrases and n-grams; a larger vocabulary captures more of that signal.

3. **A wider MLP with a lower learning rate works well.** Configs with hidden layers `(512, 256, 128)`, moderate dropout, and `lr = 5e-4` clustered at the top. The default baseline (256→64, `lr = 1e-3`) underperformed.

4. **Longer n-grams (1–4) did not help.** The `longer_ngrams` config underperformed `(1, 3)` setups, likely adding sparse/noisy features without enough gain on short ticket bodies.

5. **Accuracy vs macro F1 can disagree.** `wider_net_small_batch` had the highest validation **accuracy** (82.70%) but slightly lower **macro F1** (82.23%) than `max_features_15k`. Choosing by macro F1 favours balanced performance across all four types, including the minority `Change` class.

6. **Full training beats the 12-epoch search.** After retraining `max_features_15k` for up to 25 epochs, validation macro F1 reached **84.56%** (vs 82.60% during search). BERT still leads on validation (86.73%), but the gap is smaller than the search-only comparison suggested.

## Results: MLP vs BERT

Both models use the same dataset and stratified split (`random_state=42`). Metrics below are from actual runs.

| Model | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 | Top-3 test |
|-------|-------------:|-------------:|--------------:|--------------:|-----------:|
| TF-IDF + MLP (`max_features_15k`) | 84.17% | 84.56% | 82.14% | 82.76% | 98.28% |
| BERT (`bert-base-uncased`) | **85.43%** | **86.73%** | **82.47%** | **83.67%** | **99.96%** |

**BERT vs MLP (validation):** +1.26 pp accuracy, **+2.17 pp macro F1**.  
**BERT vs MLP (test set, 2,385 tickets):** +0.33 pp accuracy, **+0.91 pp macro F1**.

Both models generalize worse on the held-out test set than on validation. BERT’s val→test drop is larger (−3.06 pp macro F1) than the MLP’s (−1.80 pp), yet BERT still leads on test by **+0.91 pp** macro F1.

### MLP training summary

- **Config:** `max_features_15k` (15,000 TF-IDF features, hidden 512→128, lr 5e-4)
- **Hardware:** CPU
- **Training:** up to 25 epochs, early stopping at epoch 7; **best checkpoint at epoch 4**
- **Best validation:** 84.17% accuracy, **84.56% macro F1**
- **Test set:** 82.14% accuracy, **82.76% macro F1**, 98.28% top-3
- **Parameters:** ~7.8M

### MLP test set — per-class F1

| Type | Precision | Recall | F1 | Support |
|------|----------:|-------:|---:|--------:|
| Change | 0.94 | 0.93 | 0.94 | 257 |
| Request | 0.97 | 0.98 | 0.98 | 700 |
| Incident | 0.79 | 0.78 | 0.78 | 929 |
| Problem | 0.61 | 0.62 | **0.61** | 499 |

### BERT training summary

- **Model:** `bert-base-uncased` (~109M parameters)
- **Hardware:** CUDA (Colab GPU), batch size 16, max length 256
- **Training:** up to 10 epochs, early stopping at epoch 7; **best checkpoint at epoch 5**
- **Best validation:** 85.43% accuracy, **86.73% macro F1**
- **Test set:** 82.47% accuracy, **83.67% macro F1**, 99.96% top-3

The `LOAD REPORT` warnings when loading the checkpoint are **normal** for fine-tuning: pretraining heads (`cls.predictions.*`, `cls.seq_relationship.*`) are dropped as UNEXPECTED, and the classification head (`classifier.*`) is MISSING because it is **newly initialized** for this 4-class task and learned during training.

### BERT test set — per-class F1

| Type | Precision | Recall | F1 | Support |
|------|----------:|-------:|---:|--------:|
| Change | 0.96 | 0.92 | **0.94** | 257 |
| Request | 0.98 | 0.98 | **0.98** | 700 |
| Incident | 0.83 | 0.74 | 0.78 | 929 |
| Problem | 0.59 | 0.72 | **0.65** | 499 |

**Problem** remains the hardest class (lowest F1 for both precision and recall balance). **Request** and **Change** are classified most reliably. Sample predictions on hand-written tickets were correct with high confidence (e.g. crash → Incident 99.4%, documentation → Request 99.95%, recurring errors → Problem 93.9%, migration → Change 99.6%).

### Takeaways

1. **BERT beats the tuned MLP on both validation and test** — but the test-set gap is modest (+0.91 pp macro F1), while validation suggests a larger difference (+2.17 pp).
2. **Both models overfit somewhat** — validation F1 exceeds test F1 for MLP (84.56% → 82.76%) and BERT (86.73% → 83.67%).
3. **Problem is hardest for both** — MLP F1 0.61, BERT F1 0.65; BERT helps that class most but it remains the weak point.
4. **Request and Change are easy** — both models reach ~0.94–0.98 F1 on these types.
5. **Trade-off:** BERT adds ~1 pp test F1 over a well-tuned MLP, but needs a GPU and ~109M parameters vs ~7.8M on CPU.

## How to run

**MLP baseline:** open [`Customer_Support_Ticket_Classifier_TF_IDF.ipynb`](Customer_Support_Ticket_Classifier_TF_IDF.ipynb) → **Runtime → Run all** (CPU is fine).

**BERT:** open [`Customer_Support_Ticket_Classifier_BERT.ipynb`](Customer_Support_Ticket_Classifier_BERT.ipynb) → **Runtime → Change runtime type → T4 GPU** → **Run all**.

Both notebooks use the same dataset and split (`random_state=42`), so test metrics are directly comparable.


## Requirements

**MLP notebook:** `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `joblib`

**BERT notebook:** above + `transformers`
