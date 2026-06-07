"""
Mock evaluation: run the ASCII CAPTCHA scoring pipeline without API keys.

Loads challenge files from a directory, produces deterministic random responses,
and writes results in the same CSV format as the live evaluation pipeline.

This allows reviewers to verify the full pipeline offline.

Usage:
    python scripts/mock_eval.py --data-dir examples/ascii_samples/ --out results/mock/
"""
import argparse
import csv
import os
import random
import re
import string
import sys
import time


CHARSET = string.ascii_uppercase + string.digits


def random_response(label: str, correct_prob: float = 0.02) -> str:
    """Return a deterministic fake model response.

    With probability correct_prob the model 'guesses' correctly.
    Otherwise it returns a random string of the same length.
    """
    if random.random() < correct_prob:
        return label.lower()
    length = len(label)
    return "".join(random.choices(CHARSET.lower(), k=length))


def validate_output(text: str) -> str | None:
    text = text.strip()
    if not text or len(text) == 0:
        return None
    if len(text) > 25:
        return None
    if re.search(r"[.!?]", text):
        return None
    if text.startswith("[ERROR"):
        return None
    if "\n" in text or "#" in text:
        return None
    return text


def run_mock_eval(data_dir: str, out_dir: str, models: list[str], seed: int = 42):
    random.seed(seed)
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    challenge_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".txt")
    ]

    if not challenge_files:
        print(f"No .txt challenge files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(challenge_files)} challenge(s) in {data_dir}")

    for model in models:
        model_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
        csv_path = os.path.join(raw_dir, f"{model_slug}-results.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Actual Solution", "Predicted Solution", "Response Time (s)"])

            for path in challenge_files:
                label = os.path.splitext(os.path.basename(path))[0].lower()
                t0 = time.time()
                raw = random_response(label)
                elapsed = time.time() - t0 + random.uniform(0.5, 3.0)  # simulate latency

                validated = validate_output(raw)
                if validated is not None:
                    writer.writerow([label, validated, f"{elapsed:.4f}"])

        print(f"  [{model}] Results written to {csv_path}")

    print(f"\nMock evaluation complete. Raw CSVs in {raw_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Mock ASCII CAPTCHA evaluation (no API keys)")
    parser.add_argument("--data-dir", default="examples/ascii_samples", help="Directory of .txt challenge files")
    parser.add_argument("--out", default="results/mock", help="Output directory for results")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["mock/model-a", "mock/model-b"],
        help="Mock model names to simulate",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_mock_eval(args.data_dir, args.out, args.models, args.seed)


if __name__ == "__main__":
    main()
