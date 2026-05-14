# 🎸 Multi-Modal Hybrid Instrument Recommender

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-GPU-76b900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project%20Page-GitHub%20Pages-222?logo=github)](https://rajneeshbabu.github.io/multimodal-instrument-recommender/)

🌐 **[View Project Page →](https://rajneeshbabu.github.io/multimodal-instrument-recommender/)**

A hybrid neural recommender system trained on the **Amazon Musical Instruments** dataset (~10K real reviews), fusing collaborative-filtering ID embeddings with multi-modal item attributes — product text (TF-IDF + LSA), price, category, and brand.

**[🌐 Live Portfolio Page](https://rajneeshbabu.github.io/multimodal-instrument-recommender)**

---

## Architecture

```
User Tower:  [user_id_emb(32) ‖ user_attr_enc(4→64→32)]  →  fused 64-dim
Item Tower:  [item_id_emb(32) ‖ item_attr_enc(73→64→32)] →  fused 64-dim
             ────────────────────────────────────────────
             cat([user_fused, item_fused])  →  128-dim
             Interaction MLP: 128 → 64 → 32 → 1
             BatchNorm + Dropout(0.2) + sigmoid
```

**What makes it "multi-modal":**

| Modality | Source | Representation |
|---|---|---|
| Collaborative | User/Item IDs | Learned 32-dim embeddings |
| Numerical | Product price | Normalized scalar |
| Categorical | Category (15) + Brand (30) | One-hot vectors |
| **Text** | **Product titles** | **TF-IDF bigrams → TruncatedSVD(32)** |
| Behavioral | User history | avg_rating, log_reviews, recency, verified_ratio |

---

## Key Results

Evaluated on held-out test set (20% split) · Amazon Musical Instruments · 99.2% sparse · 30-epoch Adam + BPR:

| Metric | @5 | @10 | @20 |
|---|---|---|---|
| **Hit Rate** | 0.0333 | **0.0367** | 0.0433 |
| **NDCG** | 0.0151 | **0.0155** | 0.0167 |
| **Precision** | 0.0067 | 0.0037 | 0.0023 |
| **Recall** | 0.0164 | 0.0172 | 0.0211 |
| **MRR** | 0.0198 | **0.0204** | 0.0207 |

> HR@10 = 0.0367 vs. random baseline 0.011 (**3.3× better than random**)  
> NDCG@10 and MRR@10 both **doubled** vs. the 20-epoch StepLR run — thanks to CosineAnnealingLR keeping a useful LR throughout all 30 epochs.

---

## Ablation Study

| Variant | ID Embed | Attr Enc | HR@10 | NDCG@10 | Params |
|---|---|---|---|---|---|
| ID-Only *(20 ep)* | ✅ | ❌ | 0.0100 | 0.0063 | 2,781,697 |
| Attr-Only *(20 ep)* | ❌ | ✅ | 0.0000 | 0.0000 | 28,353 |
| **Full Hybrid ★** *(30 ep)* | **✅** | **✅** | **0.0367** | **0.0155** | **2,799,105** |

**Finding:** Attr-Only alone is insufficient — collaborative filtering (ID embeddings) is essential for sparse real-world data. Full Hybrid with 30 epochs significantly outperforms both ablation variants.

---

## Training Analysis

- **Phase 1 — SGD + MSE:** Loss stalls from 0.0989 → 0.0854 over epochs 5–10; 16.7% of gradient tensors show near-zero updates
- **Phase 2 — Adam + BPR:** Loss drops steeply from 0.717 → 0.115 over 30 epochs — CosineAnnealingLR avoids premature LR collapse
- **Optimizer:** Adam (lr=1e-3, weight_decay=1e-5) + CosineAnnealingLR(T_max=30, eta_min=1e-5)
- **Loss:** Bayesian Personalized Ranking — `-log σ(score_pos − score_neg)`
- **Model scale:** 2,799,105 parameters · item embedding covers full 84,901-item catalog

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/rajneeshbabu/multimodal-instrument-recommender.git
cd multimodal-instrument-recommender

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook multimodal_instrument_recommender.ipynb
```

The notebook auto-downloads the Amazon Musical Instruments dataset (~3 MB) on first run. If download fails, it falls back to a synthetic dataset.

---

## Project Structure

```
.
├── multimodal_instrument_recommender.ipynb  # Main notebook (18 cells)
├── index.html                               # Portfolio page
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

- **Source:** [Amazon Product Reviews (UCSD, 2018)](http://jmcauley.ucsd.edu/data/amazon/)
- **Subset:** Musical Instruments (5-core)
- **Scale:** 1,429 users · 84,901 items in metadata (900 with interactions) · 10,261 reviews
- **Sparsity:** 99.2% (classic cold-start scenario)

---

## Technical Highlights

1. **TF-IDF + LSA text features** — 1,000-vocab bigrams on product titles, compressed to 32-dim via TruncatedSVD (18.7% variance explained on full 84K catalog). Top semantic components capture guitar/electric types, pedal effects, and amp terms.

2. **BPR loss** — Trains on `(user, pos_item, neg_item)` triplets. Optimises pairwise ranking directly rather than per-item binary classification.

3. **Negative sampling** — Each positive interaction is paired with one uniformly sampled unobserved item per epoch, with per-user positive-set filtering to avoid false negatives.

4. **Gradient diagnosis** — SGD stagnation tracked via fraction of parameter tensors with mean |∇| < 1e-6, motivating the switch to Adam.

5. **Comprehensive evaluation** — Hit Rate, NDCG, Precision, Recall, and MRR reported at K ∈ {5, 10, 20}.

---

## License

MIT — see [LICENSE](LICENSE)
