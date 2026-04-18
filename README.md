# Gen AI Annotation Quality & Evaluation Pipeline

End-to-end pipeline for **multi-annotator text classification** on AG News, comparing simulated human annotators, an LLM annotator (Claude), and a fine-tuned DistilBERT model. Measures inter-annotator agreement, detects systematic bias, and produces a publication-ready HTML report.

## Overview

Given the same 1,000-item eval set (AG News, 4 classes — World / Sports / Business / Sci/Tech), the pipeline:

1. Simulates 5 human annotators with distinct accuracy and bias profiles.
2. Labels the same items with **Claude** via the Anthropic API as a 6th annotator.
3. Fine-tunes **DistilBERT** on the AG News train split (120K items) and scores it on the **same** 1K eval set — no leakage.
4. Computes IAA (Krippendorff's α, Fleiss' κ, pairwise Cohen's κ), per-annotator accuracy, confusion matrices, and flags systematic bias.
5. Runs a three-way comparison (human consensus vs. Claude vs. fine-tuned DistilBERT) with complementarity analysis.
6. Renders everything into a self-contained HTML report with embedded charts.

## Architecture

```
                  ┌──────────────────┐
                  │  AG News (test)  │  1,000 items, balanced
                  └────────┬─────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   Simulated 5×       Claude LLM       Fine-tuned DistilBERT
   annotators         (Anthropic)      (trained on AG News train)
         │                 │                 │
         └─────────┬───────┴─────────┬───────┘
                   ▼                 ▼
         Quality metrics      3-way evaluation
         (IAA, bias, κ)       (accuracy, F1, complementarity)
                   │                 │
                   └────────┬────────┘
                            ▼
                  HTML report + JSON outputs
```

Each stage is an independent script — outputs flow through CSV/JSON files in `data/` and `outputs/`, so any stage can be re-run in isolation.

## Key Features

- **Multi-method annotation**: simulated humans, Claude API, and fine-tuned transformer, all scored on identical inputs.
- **Inter-annotator agreement**: Krippendorff's α (overall + per category), Fleiss' κ, pairwise Cohen's κ heatmap.
- **Bias detection**: flags (true→predicted) category pairs where an annotator's error rate exceeds 2× the cohort average — validated against the injected bias profiles.
- **Complementarity analysis**: quantifies how often Claude corrects human-consensus errors (hybrid-workflow signal).
- **Train/eval discipline**: fine-tuning uses the AG News train split only; the 1K eval set is sampled from the test split and never seen at training time.
- **Self-contained report**: `outputs/evaluation_report.html` embeds all charts as base64 — no external assets.

## Tech Stack

| Layer | Tools |
|------|------|
| Language | Python 3.10+ |
| Data | `datasets` (HuggingFace), `pandas`, `numpy` |
| LLM annotation | `anthropic` (Claude Sonnet 4) |
| Fine-tuning | `transformers`, `torch`, `accelerate` (DistilBERT) |
| Metrics | `scikit-learn`, `krippendorff` |
| Reporting | `matplotlib`, `seaborn`, `jinja2` |

## Project Structure

```
gen-ai-annotation-pipeline/
├── README.md
├── requirements.txt
├── data/
│   ├── ag_news_sample.csv          # 1,000-item balanced eval set (from AG News test)
│   ├── annotations_human.csv       # + 5 simulated human annotator columns
│   └── annotations_all.csv         # + Claude column + fine-tuned column + consensus
├── src/
│   ├── 01_data_prep.py             # Download AG News, sample 250 per category
│   ├── 02_simulate_annotators.py   # 5 humans with scripted accuracy/bias profiles
│   ├── 03_llm_annotator.py         # Claude API, batched (25/request), with retries
│   ├── 04_quality_metrics.py       # IAA, confusion matrices, bias flags, ranking
│   ├── 05_evaluation.py            # 3-way accuracy/F1 + complementarity analysis
│   ├── 06_visualization.py         # Charts + Jinja2 HTML report
│   ├── 07_finetune_classifier.py   # Full fine-tune DistilBERT, evaluate on 1K
│   └── utils.py                    # Shared constants + majority-vote consensus
├── models/
│   └── distilbert_agnews/          # Saved fine-tuned model weights
├── outputs/
│   ├── evaluation_report.html      # Final report (open in browser)
│   ├── evaluation_results.json     # All metrics, 3-way outcomes, fine-tuned meta
│   ├── finetuned_results.json      # DistilBERT accuracy, per-class F1, latency
│   ├── iaa_metrics.json            # Krippendorff α + Fleiss κ
│   ├── annotator_accuracy.csv
│   ├── annotator_ranking.csv
│   ├── cohen_kappa_pairs.csv
│   └── bias_flags.csv
└── notebooks/
    └── exploration.ipynb           # Optional EDA
```

## Setup & Usage

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key (for step 3)

```bash
export ANTHROPIC_API_KEY='sk-ant-...'
```

### 3. Run the pipeline

```bash
python src/01_data_prep.py              # sample 1,000 AG News items
python src/02_simulate_annotators.py    # generate 5 human annotators
python src/03_llm_annotator.py          # Claude API annotation
python src/04_quality_metrics.py        # IAA, bias, ranking
python src/05_evaluation.py             # accuracy / F1 / complementarity
python src/06_visualization.py          # render HTML report

# Optional: add fine-tuned DistilBERT as a third method
python src/07_finetune_classifier.py
python src/05_evaluation.py             # re-run to fold in DistilBERT results
python src/06_visualization.py
```

Open `outputs/evaluation_report.html` in any browser. Steps 4–6 also run without step 3 — the pipeline gracefully skips the LLM sections when Claude annotations are absent.

## Sample Outputs

### Accuracy on the shared 1,000-item eval set

| Method | Accuracy | Notes |
|---|---|---|
| Human consensus (majority of 5) | **97.0%** | Best overall |
| Fine-tuned DistilBERT | 93.7% | 5.85 ms/sample on GPU, ~\$0 marginal cost |
| Claude (Sonnet 4) | 90.5% | Zero training, strong complementarity |
| Best individual annotator | 91.8% | Expert profile |
| Worst individual annotator | 72.2% | Noisy + biased profile |

### Inter-annotator agreement

- **Krippendorff's α** (overall): 0.594 — moderate agreement, reflecting the injected noise.
- **Fleiss' κ** (5 humans): 0.561.

### Three-way outcomes (consensus / Claude / DistilBERT)

| Outcome | Count |
|---|---|
| All three correct | 851 |
| Only consensus correct | 33 |
| Only Claude correct | 2 |
| Only fine-tuned correct | 2 |
| All three wrong | 1 |

### Complementarity

When human consensus is wrong, Claude gets it right **90.0%** of the time — strong signal for hybrid human+LLM annotation workflows.

### Bias detection

Validated on the injected profiles: annotator_5's `World → Business` (2.4×) and `Sports → World` (2.3×) drift were both correctly flagged by the pipeline.

### Hardest category

**Sci/Tech**, frequently confused with Business by both Claude (F1 0.880) and the fine-tuned model (F1 0.917), confirming this is an inherent class-boundary issue rather than a per-annotator artifact.

## Design Notes

- **Why simulated annotators?** They provide ground-truth bias profiles the pipeline must rediscover — a validation harness for the IAA and bias-detection machinery before trusting it on real, unknown error patterns.
- **Why full fine-tune over LoRA?** AG News is small-headline text; the full fine-tune converges in 2 epochs and the model is small enough that parameter-efficient tuning offers no practical win.
- **Why a single 1K eval set across methods?** Apples-to-apples comparison. The fine-tuned model is trained on the AG News train split (120K), evaluated on the 1K sampled from the test split — no overlap, reproducible via seed 42.
