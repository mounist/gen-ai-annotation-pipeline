# Gen AI Annotation Quality & Evaluation Pipeline

An end-to-end pipeline to simulate, consolidate, and evaluate multi-annotator text classification with LLM-assisted annotation. Built as a portfolio project demonstrating annotation quality analysis, inter-annotator agreement metrics, and human-vs-LLM evaluation.

## Overview

This project builds a complete annotation quality pipeline on the AG News dataset (4-class news classification: World, Sports, Business, Sci/Tech):

1. **Data Preparation** — Sample 1,000 balanced items from AG News
2. **Simulated Annotators** — 5 human annotators with realistic accuracy/bias profiles
3. **LLM Annotator** — Claude as a 6th annotator via the Anthropic API
4. **Quality Metrics** — Inter-annotator agreement (Krippendorff's alpha, Cohen's/Fleiss' kappa), confusion matrices, systematic bias detection
5. **Human vs LLM Evaluation** — Three-way comparison (individual humans, consensus, LLM) with complementarity analysis
6. **Visualization** — HTML report with charts and tables

## Key Findings

- **Human consensus accuracy: 97.0%** — majority vote across 5 annotators outperforms every individual annotator
- **LLM (Claude) accuracy: 90.5%** — competitive with the best individual human annotator (91.8%)
- **Krippendorff's alpha: 0.594** | **Fleiss' kappa: 0.561** — moderate agreement across annotators, reflecting the injected noise and bias profiles
- **LLM complementarity: 90.0%** — when human consensus is wrong, the LLM gets it right 90% of the time, suggesting strong potential for hybrid annotation workflows
- **Bias detection validated**: Annotator 5 flagged with 2 systematic bias patterns (World->Business at 2.4x average, Sports->World at 2.3x average). Annotator 4's Business/Sci-Tech confusion was absorbed into average rates due to the category pair's inherent difficulty across all annotators
- **Hardest category**: Sci/Tech — lowest F1 for both LLM (0.880) and most human annotators, frequently confused with Business

## Design Decisions

Simulated annotators serve as ground-truth validation — since we know the injected bias profiles (e.g. annotator_4's Business<->Sci/Tech confusion, annotator_5's World->Business drift), we can verify the pipeline correctly detects them. The LLM annotator (Claude) then serves as a real-world test with unknown error patterns, demonstrating the pipeline generalizes beyond synthetic inputs.

## Tech Stack

- **Python 3.10+**
- **Claude API** (Anthropic SDK) — LLM annotation
- **Krippendorff's alpha** — Inter-annotator agreement
- **scikit-learn** — Cohen's kappa, confusion matrices, F1 scores
- **matplotlib / seaborn** — Visualization
- **HuggingFace Datasets** — AG News data source
- **Jinja2** — HTML report generation

## Project Structure

```
gen-ai-annotation-pipeline/
├── README.md
├── requirements.txt
├── data/
│   ├── ag_news_sample.csv          # 1,000 sampled news items
│   ├── annotations_human.csv       # Simulated human annotations
│   └── annotations_all.csv         # Human + LLM annotations
├── src/
│   ├── 01_data_prep.py             # Download AG News, sample 1,000
│   ├── 02_simulate_annotators.py   # Simulate 5 human annotators
│   ├── 03_llm_annotator.py         # Claude API as 6th annotator
│   ├── 04_quality_metrics.py       # IAA, error rates, bias detection
│   ├── 05_evaluation.py            # Human vs LLM comparison
│   ├── 06_visualization.py         # Charts and HTML report
│   └── utils.py                    # Shared utilities (consensus, constants)
├── outputs/
│   └── evaluation_report.html      # Final visual report
└── notebooks/
    └── exploration.ipynb           # Optional EDA notebook
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the pipeline

Each script can be run independently or in sequence:

```bash
# Step 1: Download and sample data
python src/01_data_prep.py

# Step 2: Simulate human annotators
python src/02_simulate_annotators.py

# Step 3: LLM annotation (requires API key)
export ANTHROPIC_API_KEY='your-key-here'
python src/03_llm_annotator.py

# Step 4: Compute quality metrics
python src/04_quality_metrics.py

# Step 5: Human vs LLM evaluation
python src/05_evaluation.py

# Step 6: Generate report
python src/06_visualization.py
```

### 3. View the report

Open `outputs/evaluation_report.html` in your browser.

> **Note:** Steps 4-6 can run without Step 3 (LLM annotations) — they will use only the human annotator data. For the full human-vs-LLM comparison, run Step 3 first.

## Annotator Profiles

| Annotator | Accuracy | Bias |
|-----------|----------|------|
| annotator_1 | ~92% | None (expert) |
| annotator_2 | ~85% | None (good) |
| annotator_3 | ~78% | None (average) |
| annotator_4 | ~80% | Confuses Business ↔ Sci/Tech |
| annotator_5 | ~70% | World → Business, Sports → World |
