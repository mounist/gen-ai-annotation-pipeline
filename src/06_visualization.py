"""
Step 6: Visualization & HTML Report
Generate charts and an HTML evaluation report.
"""

import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score
from pathlib import Path
from jinja2 import Template
import base64
from io import BytesIO

from utils import compute_consensus, CATEGORIES, ANNOTATOR_COLS

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"


def fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 PNG for embedding in HTML."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{b64}"


def plot_accuracy_comparison(df: pd.DataFrame, has_llm: bool) -> str:
    """Bar chart: accuracy comparison across all annotators + LLM + consensus."""
    cols = ANNOTATOR_COLS[:]
    if has_llm:
        cols.append("llm_annotator")

    consensus = compute_consensus(df)

    names = []
    accs = []
    colors = []
    for col in ANNOTATOR_COLS:
        acc = (df[col] == df["true_label"]).mean()
        names.append(col.replace("annotator_", "A"))
        accs.append(acc)
        colors.append("#5B9BD5")

    cons_acc = (pd.Series(consensus) == df["true_label"]).mean()
    names.append("Consensus")
    accs.append(cons_acc)
    colors.append("#ED7D31")

    if has_llm:
        llm_acc = (df["llm_annotator"] == df["true_label"]).mean()
        names.append("LLM")
        accs.append(llm_acc)
        colors.append("#70AD47")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, accs, color=colors, edgecolor="white", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Ground Truth", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig_to_base64(fig)


def plot_kappa_heatmap(df: pd.DataFrame, has_llm: bool) -> str:
    """Heatmap of Cohen's kappa between annotator pairs."""
    from sklearn.metrics import cohen_kappa_score

    cols = ANNOTATOR_COLS[:]
    if has_llm:
        cols.append("llm_annotator")

    n = len(cols)
    kappa_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            k = cohen_kappa_score(df[cols[i]], df[cols[j]])
            kappa_matrix[i, j] = k
            kappa_matrix[j, i] = k

    labels = [c.replace("annotator_", "A").replace("llm_annotator", "LLM") for c in cols]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(kappa_matrix, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels,
                cmap="YlOrRd", vmin=0, vmax=1, ax=ax, linewidths=0.5)
    ax.set_title("Cohen's Kappa Between Annotators", fontsize=14, fontweight="bold")
    return fig_to_base64(fig)


def plot_confusion_matrices(df: pd.DataFrame, has_llm: bool) -> str:
    """Side-by-side confusion matrices for consensus and LLM."""
    consensus = compute_consensus(df)

    n_plots = 2 if has_llm else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    cm_cons = confusion_matrix(df["true_label"], consensus, labels=CATEGORIES)
    sns.heatmap(cm_cons, annot=True, fmt="d", xticklabels=CATEGORIES, yticklabels=CATEGORIES,
                cmap="Blues", ax=axes[0], linewidths=0.5)
    axes[0].set_title("Human Consensus", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    if has_llm:
        cm_llm = confusion_matrix(df["true_label"], df["llm_annotator"], labels=CATEGORIES)
        sns.heatmap(cm_llm, annot=True, fmt="d", xticklabels=CATEGORIES, yticklabels=CATEGORIES,
                    cmap="Greens", ax=axes[1], linewidths=0.5)
        axes[1].set_title("LLM (Claude)", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("True")

    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    return fig_to_base64(fig)


def plot_f1_comparison(df: pd.DataFrame, has_llm: bool) -> str:
    """Grouped bar chart of per-category F1 scores."""
    consensus = compute_consensus(df)

    cons_f1 = f1_score(df["true_label"], consensus, labels=CATEGORIES, average=None)

    x = np.arange(len(CATEGORIES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, cons_f1, width, label="Human Consensus", color="#5B9BD5", edgecolor="white")

    if has_llm:
        llm_f1 = f1_score(df["true_label"], df["llm_annotator"], labels=CATEGORIES, average=None)
        ax.bar(x + width / 2, llm_f1, width, label="LLM (Claude)", color="#70AD47", edgecolor="white")

    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Category F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig_to_base64(fig)


def plot_complementarity(df: pd.DataFrame) -> str:
    """Bar chart showing LLM corrections of consensus errors by category."""
    consensus = compute_consensus(df)
    df["_consensus"] = consensus

    cats = []
    corrections = []
    cons_errors_counts = []
    for cat in CATEGORIES:
        mask = df["true_label"] == cat
        cons_wrong = mask & (df["_consensus"] != df["true_label"])
        llm_corrects = cons_wrong & (df["llm_annotator"] == df["true_label"])
        cats.append(cat)
        cons_errors_counts.append(cons_wrong.sum())
        corrections.append(llm_corrects.sum())
    df.drop(columns=["_consensus"], inplace=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cats))
    ax.bar(x, cons_errors_counts, 0.4, label="Consensus errors", color="#FF6B6B", alpha=0.7)
    ax.bar(x, corrections, 0.4, label="LLM corrects", color="#70AD47", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Count")
    ax.set_title("LLM Corrections of Consensus Errors", fontsize=14, fontweight="bold")
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig_to_base64(fig)


def generate_html_report(charts: dict, eval_results: dict, bias_flags: list, ranking: pd.DataFrame) -> str:
    """Generate the final HTML report."""
    template = Template("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gen AI Annotation Quality & Evaluation Report</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #333; line-height: 1.6; }
        .container { max-width: 1100px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #1a1a2e; margin: 30px 0 10px; font-size: 28px; }
        .subtitle { text-align: center; color: #666; margin-bottom: 30px; font-size: 15px; }
        .section { background: white; border-radius: 12px; padding: 25px; margin-bottom: 25px;
                   box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
        .section h2 { color: #1a1a2e; border-bottom: 2px solid #5B9BD5; padding-bottom: 8px; margin-bottom: 18px; font-size: 20px; }
        .chart { text-align: center; margin: 15px 0; }
        .chart img { max-width: 100%; height: auto; border-radius: 8px; }
        .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 15px 0; }
        .metric-card { background: #f8f9fc; border-radius: 10px; padding: 18px; text-align: center; border: 1px solid #e8ecf1; }
        .metric-card .value { font-size: 28px; font-weight: bold; color: #5B9BD5; }
        .metric-card .label { font-size: 13px; color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 14px; }
        th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #e8ecf1; }
        th { background: #f8f9fc; font-weight: 600; color: #444; }
        tr:hover { background: #fafbfd; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .badge-warning { background: #fff3cd; color: #856404; }
        .badge-success { background: #d4edda; color: #155724; }
        .footer { text-align: center; color: #999; font-size: 13px; margin-top: 30px; padding: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h1>Gen AI Annotation Quality & Evaluation Report</h1>
    <p class="subtitle">Multi-annotator text classification with LLM-assisted annotation on AG News</p>

    {% if eval_results %}
    <div class="section">
        <h2>Key Metrics</h2>
        <div class="metrics-grid">
            {% if eval_results.accuracy %}
            <div class="metric-card">
                <div class="value">{{ "%.1f"|format(eval_results.accuracy.consensus * 100) }}%</div>
                <div class="label">Consensus Accuracy</div>
            </div>
            {% endif %}
            {% if eval_results.accuracy.llm_annotator is defined %}
            <div class="metric-card">
                <div class="value">{{ "%.1f"|format(eval_results.accuracy.llm_annotator * 100) }}%</div>
                <div class="label">LLM Accuracy</div>
            </div>
            {% endif %}
            {% if eval_results.complementarity_score is defined %}
            <div class="metric-card">
                <div class="value">{{ "%.1f"|format(eval_results.complementarity_score * 100) }}%</div>
                <div class="label">Complementarity</div>
            </div>
            {% endif %}
            {% if eval_results.disagreements is defined %}
            <div class="metric-card">
                <div class="value">{{ eval_results.disagreements }}</div>
                <div class="label">LLM-Consensus Disagreements</div>
            </div>
            {% endif %}
        </div>
    </div>
    {% endif %}

    <div class="section">
        <h2>1. Accuracy Comparison</h2>
        <div class="chart"><img src="{{ charts.accuracy }}" alt="Accuracy Comparison"></div>
    </div>

    <div class="section">
        <h2>2. Inter-Annotator Agreement (Cohen's Kappa)</h2>
        <div class="chart"><img src="{{ charts.kappa }}" alt="Kappa Heatmap"></div>
    </div>

    <div class="section">
        <h2>3. Confusion Matrices</h2>
        <div class="chart"><img src="{{ charts.confusion }}" alt="Confusion Matrices"></div>
    </div>

    <div class="section">
        <h2>4. Per-Category F1 Scores</h2>
        <div class="chart"><img src="{{ charts.f1 }}" alt="F1 Comparison"></div>
    </div>

    {% if charts.complementarity %}
    <div class="section">
        <h2>5. LLM Complementarity</h2>
        <p style="margin-bottom:10px;">Items where the LLM corrects human consensus errors, broken down by category.</p>
        <div class="chart"><img src="{{ charts.complementarity }}" alt="Complementarity"></div>
    </div>
    {% endif %}

    <div class="section">
        <h2>6. Annotator Ranking</h2>
        <table>
            <tr><th>Rank</th><th>Annotator</th><th>Accuracy</th><th>Avg Kappa</th><th>Consensus Agreement</th><th>Bias Flags</th><th>Composite Score</th></tr>
            {% for _, row in ranking.iterrows() %}
            <tr>
                <td>{{ loop.index }}</td>
                <td>{{ row.annotator }}</td>
                <td>{{ "%.4f"|format(row.accuracy) }}</td>
                <td>{{ "%.4f"|format(row.avg_kappa) }}</td>
                <td>{{ "%.4f"|format(row.consensus_agreement) }}</td>
                <td>{% if row.n_bias_flags > 0 %}<span class="badge badge-warning">{{ row.n_bias_flags }} flags</span>{% else %}<span class="badge badge-success">Clean</span>{% endif %}</td>
                <td><strong>{{ "%.4f"|format(row.composite_score) }}</strong></td>
            </tr>
            {% endfor %}
        </table>
    </div>

    {% if bias_flags %}
    <div class="section">
        <h2>7. Systematic Bias Detection</h2>
        <table>
            <tr><th>Annotator</th><th>True Category</th><th>Mislabeled As</th><th>Error Rate</th><th>Avg Rate</th><th>Ratio</th><th>Count</th></tr>
            {% for flag in bias_flags %}
            <tr>
                <td>{{ flag.annotator }}</td>
                <td>{{ flag.true_category }}</td>
                <td>{{ flag.predicted_category }}</td>
                <td>{{ "%.3f"|format(flag.error_rate) }}</td>
                <td>{{ "%.3f"|format(flag.avg_error_rate) }}</td>
                <td><span class="badge badge-warning">{{ "%.1f"|format(flag.ratio) }}x</span></td>
                <td>{{ flag.count }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    {% endif %}

    <div class="footer">
        Generated by Gen AI Annotation Quality & Evaluation Pipeline
    </div>
</div>
</body>
</html>
    """)

    return template.render(
        charts=charts,
        eval_results=eval_results,
        bias_flags=bias_flags,
        ranking=ranking,
    )


def main():
    all_path = DATA_DIR / "annotations_all.csv"
    human_path = DATA_DIR / "annotations_human.csv"

    if all_path.exists():
        df = pd.read_csv(all_path)
    elif human_path.exists():
        df = pd.read_csv(human_path)
    else:
        print("ERROR: No annotation file found. Run previous steps first.")
        return

    has_llm = "llm_annotator" in df.columns and df["llm_annotator"].notna().all()
    print(f"Loaded {len(df)} items (LLM annotations: {'yes' if has_llm else 'no'})")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate charts
    print("Generating charts...")
    charts = {
        "accuracy": plot_accuracy_comparison(df, has_llm),
        "kappa": plot_kappa_heatmap(df, has_llm),
        "confusion": plot_confusion_matrices(df, has_llm),
        "f1": plot_f1_comparison(df, has_llm),
        "complementarity": plot_complementarity(df) if has_llm else None,
    }

    # Load evaluation results
    eval_path = OUTPUTS_DIR / "evaluation_results.json"
    eval_results = json.loads(eval_path.read_text()) if eval_path.exists() else {}

    # Load bias flags
    bias_path = OUTPUTS_DIR / "bias_flags.csv"
    bias_flags = pd.read_csv(bias_path).to_dict("records") if bias_path.exists() else []

    # Load ranking
    ranking_path = OUTPUTS_DIR / "annotator_ranking.csv"
    ranking = pd.read_csv(ranking_path, index_col=0) if ranking_path.exists() else pd.DataFrame()

    # Generate HTML report
    print("Generating HTML report...")
    html = generate_html_report(charts, eval_results, bias_flags, ranking)

    report_path = OUTPUTS_DIR / "evaluation_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
