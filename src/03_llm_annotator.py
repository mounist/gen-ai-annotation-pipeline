"""
Step 3: LLM Annotator via Claude API
Use Claude to classify each headline, batching 25 items per API call.
"""

import json
import os
import time
import pandas as pd
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Please install anthropic: pip install anthropic")
    raise

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BATCH_SIZE = 25
MODEL = "claude-sonnet-4-20250514"
VALID_LABELS = {"World", "Sports", "Business", "Sci/Tech"}


def classify_batch(client: anthropic.Anthropic, items: list[dict]) -> dict[int, str]:
    """Send a batch of items to Claude for classification."""
    items_text = "\n".join(f'{item["item_id"]}: {item["text"]}' for item in items)

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=(
            "You are a news classification annotator. Classify each headline into exactly one "
            'category: World, Sports, Business, Sci/Tech. Return only valid JSON mapping '
            'item_id (as integer) to category string. No explanation, no markdown fences.'
        ),
        messages=[{"role": "user", "content": f"Classify these headlines:\n\n{items_text}"}],
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]

    parsed = json.loads(raw)
    # Normalize keys to int
    result = {}
    for k, v in parsed.items():
        item_id = int(k)
        label = v.strip()
        if label not in VALID_LABELS:
            print(f"  WARNING: item {item_id} got invalid label '{label}', skipping")
            continue
        result[item_id] = label
    return result


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        print("Example: export ANTHROPIC_API_KEY='sk-ant-...'")
        return

    client = anthropic.Anthropic(api_key=api_key)

    df = pd.read_csv(DATA_DIR / "annotations_human.csv")
    print(f"Loaded {len(df)} items")

    results = {}
    items = df[["item_id", "text"]].to_dict("records")
    total_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(items), BATCH_SIZE):
        batch = items[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"Processing batch {batch_num}/{total_batches} (items {batch[0]['item_id']}-{batch[-1]['item_id']})...")

        retries = 0
        while retries < 3:
            try:
                batch_results = classify_batch(client, batch)
                results.update(batch_results)
                break
            except json.JSONDecodeError as e:
                retries += 1
                print(f"  JSON parse error (attempt {retries}/3): {e}")
                if retries == 3:
                    print(f"  FAILED batch {batch_num}, skipping")
            except anthropic.APIError as e:
                retries += 1
                print(f"  API error (attempt {retries}/3): {e}")
                time.sleep(2 ** retries)
                if retries == 3:
                    print(f"  FAILED batch {batch_num}, skipping")

        time.sleep(0.5)  # Rate limiting

    # Map results back to dataframe
    df["llm_annotator"] = df["item_id"].map(results)
    missing = df["llm_annotator"].isna().sum()
    if missing > 0:
        print(f"\nWARNING: {missing} items missing LLM annotations")

    # Compute accuracy
    valid = df.dropna(subset=["llm_annotator"])
    accuracy = (valid["llm_annotator"] == valid["true_label"]).mean()
    print(f"\nLLM Accuracy vs ground truth: {accuracy:.4f} ({len(valid)} items)")

    # Per-category accuracy
    for cat in VALID_LABELS:
        mask = valid["true_label"] == cat
        if mask.sum() > 0:
            cat_acc = (valid.loc[mask, "llm_annotator"] == cat).mean()
            print(f"  {cat}: {cat_acc:.4f}")

    out_path = DATA_DIR / "annotations_all.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
