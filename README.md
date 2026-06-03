# Customer Support Ticket Classifier

Classify IT support tickets into one of four types — **Incident**, **Request**, **Problem**, or **Change** — using only the ticket **body** text.

Built as a hands-on deep learning project: a classic **TF-IDF + PyTorch MLP** pipeline with hyperparameter search, trained on English support tickets and runnable in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/boba1987/advanced-neural-networks-and-deep-learning/blob/main/Customer_Support_Ticket_Classifier.ipynb)

## What this project does

Support teams tag every ticket with a type. That routing step is repetitive and slow at scale. This project trains a model to predict the type from the ticket message alone.

**Input:** ticket body (`body` column)  
**Output:** one of `Incident`, `Request`, `Problem`, `Change`

The notebook walks through the full workflow:

1. Load and explore the dataset  
2. Clean text, encode labels, split train / validation / test (80% / 10% / 20%, stratified)  
3. Build TF-IDF + MLP using the **best config from hyperparameter search** (`max_features_15k`)  
4. Train with early stopping (validation **macro F1**)  
5. Evaluate on the held-out test set (accuracy, macro F1, confusion matrix)  
6. Save the model and run sample predictions  

All six candidate configs are defined at the top of Phase 3 for reference; the notebook skips re-running the search on every execution.

No subject line, queue, priority, or agent reply is used — only the customer message.

## Dataset

| File | Rows | Columns |
|------|------|---------|
| `dataset-tickets-en.csv` | 11,921 | `body`, `type` |

The raw export included subject, answer, tags, queue, and other fields. Those were removed so the file only contains what the model needs (~4.4 MB, down from ~10 MB).

**Class distribution:**

| Type | Count |
|------|------:|
| Incident | 4,642 |
| Request | 3,498 |
| Problem | 2,498 |
| Change | 1,285 |

`Change` is the smallest class, so **macro F1** (average F1 across classes) is a better tuning metric than accuracy alone.

## What we were trying to do

The goal was to build a **strong baseline classifier** without transformers:

- Use a simple, interpretable stack: **TF-IDF bag-of-ngrams → feedforward neural network**
- Tune it properly instead of guessing hyperparameters
- Compare six configurations in a structured search, each trained for up to 12 epochs with early stopping
- Select the best setup by **validation macro F1**, then retrain for the full epoch budget

This answers a practical question: *how much performance can you get from a lightweight model on short support-ticket text, and which knobs matter most?*

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

6. **~82–83% validation performance is a solid TF-IDF ceiling.** Further gains may need richer models (e.g. fine-tuned transformers), better text features, or using additional fields (subject, tags) — at the cost of complexity.

## How to run

1. Open [`Customer_Support_Ticket_Classifier.ipynb`](Customer_Support_Ticket_Classifier.ipynb) in Colab (badge above) or locally.  
2. **Runtime → Run all** — CPU is fine; full training takes a few minutes.  
3. Artifacts are written to the working directory (gitignored): `ticket_classifier.pth`, `tfidf_vectorizer.pkl`, `label_encoder.pkl`.

## Project structure

```
├── Customer_Support_Ticket_Classifier.ipynb   # Full pipeline
├── dataset-tickets-en.csv                     # Cleaned training data
├── README.md
└── .gitignore                                 # Model artifacts, checkpoints
```

## Requirements

Installed in the notebook:

- `torch`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `joblib`
