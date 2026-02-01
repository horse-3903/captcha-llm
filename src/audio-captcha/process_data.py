import os
import re
import pandas as pd
from tqdm import tqdm

def tidy_results(parent_path: str):
    """
    Cleans all raw model CSVs in the given parent directory and produces:
      1. Cleaned per-model CSVs under `clean/`
      2. A combined summary.csv with mean accuracy (Result) per model
    """
    raw_path = os.path.join(parent_path, "raw")
    clean_path = os.path.join(parent_path, "clean")
    summary_path = os.path.join(parent_path, "summary.csv")

    assert os.path.exists(raw_path), f"Raw path not found: {raw_path}"
    os.makedirs(clean_path, exist_ok=True)

    summaries = []

    files = [f for f in os.listdir(raw_path) if f.endswith(".csv")]
    for file in files:
        print(f"🧹 Cleaning {file}...")

        file_path = os.path.join(raw_path, file)
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")
            continue

        # --- Basic sanitization ---
        df = df.dropna(how="all")
        df = df.astype(str).apply(lambda col: col.str.replace(r"[^\x00-\x7F]", "", regex=True))
        df.columns = [c.strip() for c in df.columns]

        # --- Determine Actual & Predicted columns ---
        possible_cols = [c.lower() for c in df.columns]
        if "predicted" in possible_cols and "correct" in possible_cols:
            actual_col = df.columns[possible_cols.index("correct")]
            pred_col = df.columns[possible_cols.index("predicted")]
        elif len(df.columns) >= 2:
            actual_col, pred_col = df.columns[0], df.columns[1]
        else:
            print(f"⚠️ Skipping {file}: no valid columns found.")
            continue

        # --- Fix messy predictions ---
        def clean_pred(val: str):
            val = str(val).strip()
            # If multiple lines or sentences, extract last number
            if "\n" in val or "." in val or len(val) > 3:
                matches = re.findall(r"\b\d+\b", val)
                return matches[-1] if matches else ""
            return val  # if already clean like "1", "2", or "3"

        df[pred_col] = df[pred_col].apply(clean_pred)

        # --- Compute correctness ---
        df["Result"] = (df[pred_col].astype(str).str.strip() == df[actual_col].astype(str).str.strip()).astype(int)

        # --- Save cleaned file ---
        cleaned_path = os.path.join(clean_path, file)
        df.to_csv(cleaned_path, index=False)
        print(f"✅ Saved cleaned: {cleaned_path}")

        # --- Summarize performance ---
        acc = df["Result"].mean() if not df.empty else 0.0
        summaries.append({
            "Model": file.split("-results")[0],
            "Result": round(acc, 4),
            "Count": len(df)
        })

    # --- Write combined summary ---
    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.sort_values(by="Result", ascending=False, inplace=True)
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Summary written to {summary_path}")
        print(summary_df)
    else:
        print("⚠️ No valid files found; summary not generated.")


def main():
    parent_path = "results/audio-test-2"
    
    for path in os.listdir(parent_path):
        tidy_results(os.path.join(parent_path, path))

if __name__ == "__main__":
    main()
