"""
Score ASCII CAPTCHA result CSVs.

Metrics match those reported in the paper (arXiv:2604.03612):
  - Full accuracy: fraction of exact matches (what is required to pass the CAPTCHA)
  - Text similarity: normalised Levenshtein ratio via difflib.SequenceMatcher,
    measuring how close a model's output is to the ground truth even when not exact
  - Mean response time: average per-query wall-clock time in seconds

Usage:
    python scripts/score_ascii_results.py --results-dir results/mock/raw/ --out results/mock/summary.csv
"""
import argparse
import csv
import difflib
import os
import sys


def text_similarity(actual: str, predicted: str) -> float:
    """Normalised Levenshtein ratio via difflib.SequenceMatcher (paper metric)."""
    return difflib.SequenceMatcher(None, actual, predicted).ratio()


def score_file(csv_path: str) -> dict:
    exact_matches = 0
    similarities = []
    response_times = []
    total = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            actual = row.get("Actual Solution", "").strip().lower()
            predicted = row.get("Predicted Solution", "").strip().lower()
            if not actual:
                continue
            total += 1
            if actual == predicted:
                exact_matches += 1
            similarities.append(text_similarity(actual, predicted))
            t = row.get("Response Time (s)", "")
            if t:
                try:
                    response_times.append(float(t))
                except ValueError:
                    pass

    if total == 0:
        return {"full_accuracy": 0.0, "text_similarity": 0.0, "avg_response_time": None, "count": 0}

    return {
        "full_accuracy": round(exact_matches / total, 4),
        "text_similarity": round(sum(similarities) / len(similarities), 4),
        "avg_response_time": round(sum(response_times) / len(response_times), 4) if response_times else None,
        "count": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Score ASCII CAPTCHA results (paper metrics)")
    parser.add_argument("--results-dir", required=True, help="Directory containing per-model result CSVs")
    parser.add_argument("--out", required=True, help="Output path for summary CSV")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        print(f"Results directory not found: {args.results_dir}")
        sys.exit(1)

    csv_files = [f for f in os.listdir(args.results_dir) if f.endswith(".csv")]
    if not csv_files:
        print(f"No CSV files found in {args.results_dir}")
        sys.exit(1)

    rows = []
    for fname in sorted(csv_files):
        model_name = fname.replace("-results.csv", "")
        fpath = os.path.join(args.results_dir, fname)
        stats = score_file(fpath)
        rows.append({"Model": model_name, **stats})
        t = f"{stats['avg_response_time']:.4f}s" if stats["avg_response_time"] is not None else "N/A"
        print(
            f"  {model_name}: "
            f"accuracy={stats['full_accuracy']:.4f}  "
            f"similarity={stats['text_similarity']:.4f}  "
            f"avg_time={t}  "
            f"n={stats['count']}"
        )

    rows.sort(key=lambda r: r["full_accuracy"], reverse=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Model", "full_accuracy", "text_similarity", "avg_response_time", "count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSummary written to {args.out}")


if __name__ == "__main__":
    main()
