import argparse
import csv
import importlib.util
import os
import random
import time
import sys
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pyfiglet


def _load_module(module_name: str, rel_path: str):
    base_dir = Path(__file__).resolve().parents[1]
    module_path = base_dir / rel_path
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


_audio_util = _load_module("audio_util", os.path.join("audio-captcha", "util.py"))
_ascii_util = _load_module("ascii_util", os.path.join("ascii-captcha", "util.py"))
_fine_tune = _load_module("fine_tune_generate", os.path.join("fine-tune", "generate_data.py"))

AudioCaptchaTester = _audio_util.AudioCaptchaTester
add_background_noise = _audio_util.add_background_noise
add_gaussian_noise = _audio_util.add_gaussian_noise
combine_audio_files = _audio_util.combine_audio_files
_text_to_image_cached = _fine_tune._text_to_image_cached
ASCIICaptchaTester = _ascii_util.ASCIICaptchaTester


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


def _write_summary_csv(out_path: str, rows: List[Dict[str, str]]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "category",
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


def time_ascii_generation(samples: int, font_path: str, font_size: int) -> Dict[str, float]:
    timings: List[float] = []
    font_cache = {}

    for _ in range(samples):
        length = random.randint(7, 15)
        captcha = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=length))
        ascii_art = pyfiglet.figlet_format(captcha, font="ascii___")

        start = time.perf_counter()
        _ = _text_to_image_cached(ascii_art, font_path, font_size, font_cache)
        timings.append(time.perf_counter() - start)

    return _summarize_timings(timings)


def _load_audio_files(csv_path: str, sample_limit: int) -> List[str]:
    df = pd.read_csv(csv_path)
    if "filename" not in df.columns:
        raise ValueError(f"CSV missing filename column: {csv_path}")
    files = df["filename"].dropna().astype(str).tolist()
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


def time_audio_generation(
    files: List[str],
    background_file: str,
    gaussian_level: float,
    combined_ratios: List[float],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    def record(mode: str, timings: List[float]):
        results[mode] = _summarize_timings(timings)

    timings_none: List[float] = []
    timings_gaussian: List[float] = []
    timings_background: List[float] = []
    timings_combined: List[float] = []

    for path in files:
        others = [p for p in files if p != path]
        if len(others) >= 2:
            mix_sources = random.sample(others, 2)
        else:
            mix_sources = others

        start = time.perf_counter()
        with open(path, "rb") as f:
            _ = f.read()
        timings_none.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = add_gaussian_noise(path, noise_level=gaussian_level)
        timings_gaussian.append(time.perf_counter() - start)

        start = time.perf_counter()
        _ = add_background_noise(path, background_file, boost=1.0)
        timings_background.append(time.perf_counter() - start)

        if mix_sources:
            start = time.perf_counter()
            _ = combine_audio_files(path, mix_sources, ratios=combined_ratios[: len(mix_sources)])
            timings_combined.append(time.perf_counter() - start)

    record("none", timings_none)
    record("gaussian", timings_gaussian)
    record("background", timings_background)
    record("combined", timings_combined)
    return results


def time_audio_tts_generation(
    csv_path: str,
    sample_limit: int,
    output_dir: str,
    model_name: str,
    use_gpu: bool,
    speaker_wav: str | None,
) -> Dict[str, float]:
    try:
        from TTS.api import TTS
    except Exception as exc:
        raise RuntimeError(f"TTS library not available: {exc}") from exc

    df = pd.read_csv(csv_path)
    if "question" not in df.columns:
        raise ValueError(f"CSV missing question column: {csv_path}")
    questions = df["question"].dropna().astype(str).tolist()
    if not questions:
        raise ValueError("No question text found in audio CSV.")

    if len(questions) < sample_limit:
        sample_limit = len(questions)

    samples = random.sample(questions, sample_limit)

    os.makedirs(output_dir, exist_ok=True)
    tts = TTS(model_name, gpu=use_gpu)

    timings: List[float] = []
    for idx, text in enumerate(samples):
        out_path = os.path.join(output_dir, f"tts_sample_{idx+1:03d}.wav")
        start = time.perf_counter()
        tts.tts_to_file(text=text, file_path=out_path, language="en", speaker_wav=speaker_wav)
        timings.append(time.perf_counter() - start)

    return _summarize_timings(timings)


def run_audio_inference(
    csv_path: str,
    results_dir: str,
    sample_limit: int,
    models: List[str],
    mode: str,
    background_file: str,
    gaussian_level: float,
    background_boost: float,
    combined_ratios: List[float],
):
    tester = AudioCaptchaTester(
        csv_path=csv_path,
        models=models,
        results_dir=results_dir,
        mode=mode,
        sample_limit=sample_limit,
        background_file=background_file,
        gaussian_level=gaussian_level,
        background_boost=background_boost,
        combined_ratios=combined_ratios,
        concurrency_limit=5,
    )
    tester.start()
    timing_path = os.path.join(results_dir, "timings_summary.csv")
    tester.write_timings_summary(timing_path)


def run_ascii_inference(
    data_path: str,
    results_dir: str,
    sample_limit: int,
    models: List[str],
    render_as_image: bool,
):
    tester = ASCIICaptchaTester(
        model_list=models,
        result_path=results_dir,
        data_path=data_path,
        concurrency_limit=5,
    )
    prompt = (
        "You are taking part in a research experiment on text pattern recognition.\n"
        "Extract and return only the exact sequence of visible alphanumeric characters you can read.\n"
        "Output only the recognized text as a single continuous string."
    )
    import asyncio

    asyncio.run(
        tester.run_all_models(
            n=sample_limit,
            prompt=prompt,
            render_as_image=render_as_image,
            report_interval=10,
        )
    )
    timing_path = os.path.join(results_dir, "timings_summary.csv")
    tester.write_timings_summary(timing_path)


def main():
    parser = argparse.ArgumentParser(description="Collect timing stats for CAPTCHA generation and audio inference")
    parser.add_argument("--ascii-samples", type=int, default=50)
    parser.add_argument("--audio-samples", type=int, default=20)
    parser.add_argument("--run-audio-inference", action="store_true")
    parser.add_argument("--run-ascii-inference", action="store_true")
    parser.add_argument("--run-audio-tts-generation", action="store_true")
    parser.add_argument("--output-dir", default="results/benchmarks")
    parser.add_argument("--audio-csv", default="data/audio-captcha/extended.csv")
    parser.add_argument("--ascii-data", default="data/ascii-captcha")
    parser.add_argument("--tts-model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    parser.add_argument("--tts-gpu", action="store_true", help="Use GPU for TTS generation")
    parser.add_argument("--tts-speaker-wav", default="", help="Path to a speaker wav for multi-speaker models")
    parser.add_argument("--background-file", default="data/audio-captcha/bg-24k.wav")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    audio_csv = Path(args.audio_csv)
    if not audio_csv.is_absolute():
        audio_csv = root / audio_csv
    background_file = Path(args.background_file)
    if not background_file.is_absolute():
        background_file = root / background_file

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    ascii_data = Path(args.ascii_data)
    if not ascii_data.is_absolute():
        ascii_data = root / ascii_data

    summary_rows: List[Dict[str, str]] = []

    font_path = str(root / "fonts" / "courier.ttf")
    font_size = 16
    ascii_stats = time_ascii_generation(args.ascii_samples, font_path, font_size)
    summary_rows.append(
        {
            "category": "ascii_generation",
            "name": "ascii_figlet_image",
            "samples": str(ascii_stats["count"]),
            "total_s": f"{ascii_stats['total_s']:.4f}",
            "mean_s": f"{ascii_stats['mean_s']:.4f}",
            "median_s": f"{ascii_stats['median_s']:.4f}",
            "p95_s": f"{ascii_stats['p95_s']:.4f}",
            "min_s": f"{ascii_stats['min_s']:.4f}",
            "max_s": f"{ascii_stats['max_s']:.4f}",
        }
    )

    audio_files = _load_audio_files(str(audio_csv), args.audio_samples)
    audio_stats = time_audio_generation(
        audio_files,
        str(background_file),
        gaussian_level=1.5,
        combined_ratios=[0.6, 0.6],
    )
    for mode, stats in audio_stats.items():
        summary_rows.append(
            {
                "category": "audio_generation",
                "name": mode,
                "samples": str(stats["count"]),
                "total_s": f"{stats['total_s']:.4f}",
                "mean_s": f"{stats['mean_s']:.4f}",
                "median_s": f"{stats['median_s']:.4f}",
                "p95_s": f"{stats['p95_s']:.4f}",
                "min_s": f"{stats['min_s']:.4f}",
                "max_s": f"{stats['max_s']:.4f}",
            }
        )

    if args.run_audio_tts_generation:
        speaker_wav = args.tts_speaker_wav.strip() or ""
        if not speaker_wav:
            candidate_files = _load_audio_files(str(audio_csv), min(args.audio_samples, 5))
            speaker_wav = candidate_files[0] if candidate_files else ""
        if not speaker_wav:
            raise ValueError("No speaker wav available for TTS generation.")
        tts_out_dir = os.path.join(str(output_dir), "audio_tts_samples")
        tts_stats = time_audio_tts_generation(
            csv_path=str(audio_csv),
            sample_limit=args.audio_samples,
            output_dir=tts_out_dir,
            model_name=args.tts_model,
            use_gpu=args.tts_gpu,
            speaker_wav=speaker_wav,
        )
        summary_rows.append(
            {
                "category": "audio_generation",
                "name": f"tts:{args.tts_model}",
                "samples": str(tts_stats["count"]),
                "total_s": f"{tts_stats['total_s']:.4f}",
                "mean_s": f"{tts_stats['mean_s']:.4f}",
                "median_s": f"{tts_stats['median_s']:.4f}",
                "p95_s": f"{tts_stats['p95_s']:.4f}",
                "min_s": f"{tts_stats['min_s']:.4f}",
                "max_s": f"{tts_stats['max_s']:.4f}",
            }
        )

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(str(output_dir), "captcha_generation_times.csv")
    _write_summary_csv(summary_path, summary_rows)

    if args.run_audio_inference:
        if "OPENROUTER_API_KEY" not in os.environ:
            print("OPENROUTER_API_KEY not set; skipping audio inference timing.")
            return
        inference_dir = os.path.join(str(output_dir), "audio_inference")
        run_audio_inference(
            csv_path=str(audio_csv),
            results_dir=inference_dir,
            sample_limit=min(args.audio_samples, 10),
            models=[
                "google/gemini-3-flash-preview",
                "openai/gpt-audio-mini",
            ],
            mode="none",
            background_file=str(background_file),
            gaussian_level=1.5,
            background_boost=4.0,
            combined_ratios=[0.6, 0.6],
        )

    if args.run_ascii_inference:
        if "OPENROUTER_API_KEY" not in os.environ:
            print("OPENROUTER_API_KEY not set; skipping ascii inference timing.")
            return
        inference_dir = os.path.join(str(output_dir), "ascii_inference")
        run_ascii_inference(
            data_path=str(ascii_data),
            results_dir=inference_dir,
            sample_limit=min(args.ascii_samples, 10),
            models=[
                "qwen/qwen3-vl-8b-instruct",
                "google/gemini-2.5-pro",
                "anthropic/claude-haiku-4.5",
            ],
            render_as_image=True,
        )


if __name__ == "__main__":
    main()
