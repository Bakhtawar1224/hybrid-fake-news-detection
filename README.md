# HybridDistilBERT-DistilGPT2 — Fake News Detection

> **98.06% accuracy** on WELFake | Hybrid transformer architecture | Adversarial GPT-2 augmentation  
> Generative AI Project — FAST NUCES — April 2026

---

## Results at a Glance

| Metric | Score |
|---|---|
| Test Accuracy | **98.06%** |
| Macro F1-Score | **0.9806** |
| Fake Precision / Recall | 0.9822 / 0.9777 |
| Real Precision / Recall | 0.9790 / 0.9833 |
| Test Set Size | 7,210 articles |

### vs Published Baselines on WELFake

| Model | Accuracy |
|---|---|
| SVM + TF-IDF | 88.30% |
| LSTM + GloVe | 91.47% |
| BERT-base fine-tuned | 96.40% |
| BERT + FastText + XAI | 97.10% |
| BERT multimodal | 97.83% |
| **HybridDistilBERT-DistilGPT2 (ours)** | **98.06%** |

---

## What This Project Does

This project proposes a **novel hybrid fake news detection architecture** that fuses two transformer encoders:

- **DistilBERT** — extracts bidirectional semantic representations via the `[CLS]` token (classification strength)
- **DistilGPT2** — extracts autoregressive fluency patterns via masked mean-pooling (robustness to LLM-generated text)

Both 768-d vectors are concatenated into a 1,536-d fused representation and passed through a fusion MLP for binary classification (fake / real).

The key novelty is **adversarial data augmentation**: DistilGPT2 generates 200 synthetic fake news articles during training, explicitly exposing the classifier to LLM-style misinformation patterns that standard BERT detectors fail on.

---

## Architecture

```
Input article (headline × 3 + body, truncated to 128 tokens)
        │
        ├─── DistilBERT (6 layers, 66M params)
        │         └── [CLS] hidden state → 768-d vector
        │
        └─── DistilGPT2 (6 layers, 82M params)
                  └── masked mean-pool  → 768-d vector
                              │
                     Concatenate → 1,536-d
                              │
                     Linear(1536 → 256)
                              │
                     LayerNorm + GELU + Dropout(0.30)
                              │
                     Linear(256 → 2) → Fake / Real
```

---

## Dataset

**WELFake** — Verma et al., IEEE Trans. Comput. Soc. Syst., 2021  
Merged from 4 sources: Kaggle, McIntire, Reuters, BuzzFeed Political

| Split | Samples | Fake | Real |
|---|---|---|---|
| Train (subset) | 20,000 | ~10,285 | ~9,715 |
| Validation | ~7,214 | ~3,711 | ~3,503 |
| Test | 7,210 | 3,503 | 3,707 |

Download from Kaggle:
```bash
kaggle datasets download -d saurabhshahane/fake-news-classification
```
Or directly from [Zenodo](https://zenodo.org/records/4561253).

---

## Repository Structure

```
├── fake_news_fast.py          # Main training script (Colab-optimised)
├── outputs/
│   ├── fig1_class_distribution.png
│   ├── fig2_length_distribution.png
│   ├── fig3_training_curves.png
│   ├── fig4_confusion_matrix.png
│   ├── architecture_diagram.png
│   └── best_model.pt          # Saved checkpoint (not tracked by git)
├── paper/
│   ├── intro_litreview.tex
│   ├── methodology_section.tex
│   ├── results_discussion_conclusion.tex
│   └── architecture_diagram.png
└── README.md
```

---

## How to Run

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

Set runtime: **Runtime → Change runtime type → T4 GPU**

### 2. Install dependencies
```python
!pip install transformers==4.40.0 scikit-learn matplotlib seaborn -q
```

### 3. Get the dataset
```python
!pip install kaggle -q
# Upload kaggle.json via the Colab files panel, then:
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d saurabhshahane/fake-news-classification -q
!unzip -q fake-news-classification.zip
```

### 4. Run the script
```python
exec(open("fake_news_fast.py").read())
```

**Expected total time: ~20–25 minutes on a free T4 GPU**

---

## Key Hyperparameters

| Parameter | Value |
|---|---|
| BERT model | `distilbert-base-uncased` |
| GPT2 model | `distilgpt2` |
| Max sequence length | 128 tokens |
| Headline repeat factor | 3× |
| Training subset | 20,000 samples |
| Batch size | 32 |
| Learning rate | 2e-5 (linear warmup) |
| Warmup steps | 100 |
| Dropout | 0.30 |
| Weight decay | 0.01 |
| Epochs | 3 (early stopping patience=2) |
| Augmentation samples | 200 |
| Mixed precision | FP16 (AMP) |

---

## Research Paper

The full IEEE conference paper is written in LaTeX using the `IEEEtran` template.  
Compile on [Overleaf](https://overleaf.com) — paste all three `.tex` files in order:

1. `intro_litreview.tex` — Introduction + Literature Review
2. `methodology_section.tex` — Dataset, Preprocessing, Architecture, Training
3. `results_discussion_conclusion.tex` — Results, Discussion, Conclusion

### Key References
- Raza et al. (2024) — BERT vs LLM fake news comparison. arXiv:2412.14276
- Verma et al. (2021) — WELFake dataset. IEEE Trans. Comput. Soc. Syst.
- Aldhaheri et al. (2024) — GBERT hybrid architecture. Heliyon, PMC
- Hamid et al. (2024) — LLMs as detectors. arXiv:2409.17416
- Suhaib et al. (2025) — BERT multimodal. Computers, MDPI

---

## Requirements

```
torch>=2.0
transformers==4.40.0
scikit-learn
matplotlib
seaborn
pandas
numpy
```

---
