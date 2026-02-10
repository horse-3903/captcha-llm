import argparse
import csv
import os
import random
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
from scipy.io import wavfile
from io import BytesIO


# ==========================================================
# Audio Utilities (copied from src/audio-captcha/util.py)
# ==========================================================

def add_gaussian_noise(input_file: str, noise_level: float = 0.1) -> bytes:
    rate, data = wavfile.read(input_file)
    if data.dtype == np.int16:
        sig = data.astype(np.float32) / 32768.0
    else:
        sig = data.astype(np.float32)

    sig_rms = np.sqrt(np.mean(sig ** 2)) + 1e-12
    noise = np.random.normal(0.0, 1.0, size=sig.shape).astype(np.float32)
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-12) * (noise_level * sig_rms)

    mixed = np.clip(sig + noise, -1.0, 1.0)
    out = BytesIO()
    wavfile.write(out, rate, (mixed * 32767.0).astype(np.int16))
    return out.getvalue()


def add_background_noise(input_file: str, background_file: str, boost: float = 1.0) -> bytes:
    rate1, data1 = wavfile.read(input_file)
    rate2, data2 = wavfile.read(background_file)
    if rate1 != rate2:
        raise ValueError("Sample rates of the input and background audio do not match.")

    min_len = min(len(data1), len(data2))
    start = random.randint(0, max(0, len(data2) - min_len))
    data2 = data2[start:start + min_len].astype(np.float32)
    data1 = data1[:min_len].astype(np.float32)

    mixed = np.clip(data1 + data2 * boost, -32768, 32767).astype(np.int16)
    out = BytesIO()
    wavfile.write(out, rate1, mixed)
    return out.getvalue()


def combine_audio_files(base_file: str, others: list[str], ratios: list[float]) -> bytes:
    rate, base = wavfile.read(base_file)
    base = base.astype(np.float32)
    mixed = base.copy()

    for i, path in enumerate(others):
        r2, d2 = wavfile.read(path)
        if r2 != rate:
            raise ValueError("Sample rate mismatch.")
        d2 = d2.astype(np.float32)[: len(mixed)]
        mixed[: len(d2)] += d2 * ratios[i]
    mixed = np.clip(mixed, -32768, 32767).astype(np.int16)
    out = BytesIO()
    wavfile.write(out, rate, mixed)
    return out.getvalue()


# ==========================================================
# Benchmarking
# ==========================================================

def _summarize_timings(timings: List[float]) -> Dict[str, float]:
    arr = np.array(timings, dtype=float)
    return {
        "count": int(len(arr)),
        "total_s": float(arr.sum()) if len(arr) else 0.0,
        "mean_s": float(arr.mean()) if len(arr) else 0.0,
        "median_s": float(np.median(arr)) if len(arr) else 0.0,
        "p95_s": float(np.percentile(arr, 95)) if len(arr) else 0.0,
        "min_s": float(arr.min()) if len(arr) else 0.0,
        "max_s": float(arr.max()) if len(arr) else 0.0,
    }


def _load_audio_files(csv_path: str, sample_limit: int) -> List[str]:
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError(f"CSV missing filename column: {csv_path}")
    files = df["filename"].dropna().astype(str).tolist()
    if not files:
        raise ValueError("No filenames found in CSV.")

    if len(files) < sample_limit:
        sample_limit = len(files)

    root = Path(__file__).resolve().parents[2]
    sampled = random.sample(files, sample_limit)
    resolved = []
    for path in sampled:
        p = Path(path)
        if not p.is_absolute():
            p = root / p
        resolved.append(str(p))
    return resolved


def time_audio_noise_ops(
    files: List[str],
    background_file: str,
    gaussian_level: float,
    combined_ratios: List[float],
    background_boost: float,
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    def record(mode: str, timings: List[float]):
        results[mode] = _summarize_timings(timings)

    timings_gaussian: List[float] = []
    timings_background: List[float] = []
    timings_combined: List[float] = []

    for path in files:
        start = time.perf_counter()
        _ = add_gaussian_noise(path, noise_level=gaussian_level)
        timings_gaussian.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = add_background_noise(path, background_file, boost=background_boost)
        timings_background.append(time.perf_counter() - start)

        others = [p for p in files if p != path]
        if len(others) >= 2:
            mix_sources = random.sample(others, 2)
        else:
            mix_sources = others

        if mix_sources:
            ratios = combined_ratios[: len(mix_sources)]
            start = time.perf_counter()
            _ = combine_audio_files(path, mix_sources, ratios=ratios)
            timings_combined.append(time.perf_counter() - start)

    record("gaussian", timings_gaussian)
    record("background", timings_background)
    record("combined", timings_combined)
    return results


def _write_summary_csv(out_path: str, rows: List[Dict[str, str]]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "samples",
                "total_s",
                "mean_s",
                "median_s",
                "p95_s",
                "min_s",
                "max_s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Benchmark audio noise/combine operations")
    parser.add_argument("--audio-csv", default="data/audio-captcha/extended.csv")
    parser.add_argument("--background-file", default="data/audio-captcha/bg-24k.wav")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--gaussian-level", type=float, default=0.7)
    parser.add_argument("--background-boost", type=float, default=1.0)
    parser.add_argument("--combined-ratios", default="0.4,0.3")
    parser.add_argument("--output", default="results/benchmarks/audio_noise_timings.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    audio_csv = Path(args.audio_csv)
    if not audio_csv.is_absolute():
        audio_csv = root / audio_csv

    background_file = Path(args.background_file)
    if not background_file.is_absolute():
        background_file = root / background_file

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    ratios = [float(x.strip()) for x in args.combined_ratios.split(",") if x.strip()]
    if not ratios:
        ratios = [0.4, 0.3]

    files = _load_audio_files(str(audio_csv), args.samples)
    stats = time_audio_noise_ops(
        files=files,
        background_file=str(background_file),
        gaussian_level=args.gaussian_level,
        combined_ratios=ratios,
        background_boost=args.background_boost,
    )

    rows: List[Dict[str, str]] = []
    for name, s in stats.items():
        rows.append(
            {
                "name": name,
                "samples": str(s["count"]),
                "total_s": f"{s['total_s']:.4f}",
                "mean_s": f"{s['mean_s']:.4f}",
                "median_s": f"{s['median_s']:.4f}",
                "p95_s": f"{s['p95_s']:.4f}",
                "min_s": f"{s['min_s']:.4f}",
                "max_s": f"{s['max_s']:.4f}",
            }
        )

    _write_summary_csv(str(output), rows)
    print(f"Wrote timings to {output}")


if __name__ == "__main__":
    main()
