"""
TODO: Regenerate paper tables from raw result files.

This script is a placeholder. To use it, place raw result files under:
  results/ascii-final-1/{image|text}/raw/    (per-model ASCII result CSVs)
  results/audio-final-2/{mode}/raw/          (per-model audio result CSVs)

Then run:
    python scripts/make_tables.py --results results/ --out paper_tables/

Expected output:
  paper_tables/ascii_image_results.csv
  paper_tables/ascii_text_results.csv
  paper_tables/audio_results_by_mode.csv

See README.md § Reproducing the Paper Results for the expected CSV schemas.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Regenerate paper tables (TODO)")
    parser.add_argument("--results", default="results/", help="Root results directory")
    parser.add_argument("--out", default="paper_tables/", help="Output directory for tables")
    args = parser.parse_args()

    print("make_tables.py: this script is a TODO placeholder.")
    print(f"  Expected results root: {os.path.abspath(args.results)}")
    print(f"  Expected output dir:   {os.path.abspath(args.out)}")
    print()
    print("To regenerate tables:")
    print("  1. Run the ASCII evaluation:  python src/ascii-captcha/main.py")
    print("  2. Run the audio evaluation:  python src/audio-captcha/main.py")
    print("  3. Aggregate audio results:   python src/audio-captcha/process_data.py")
    print("  4. Score ASCII results:       python scripts/score_ascii_results.py \\")
    print("         --results-dir results/ascii-final-1/image/raw/ \\")
    print("         --out paper_tables/ascii_image_results.csv")
    sys.exit(0)


if __name__ == "__main__":
    main()
