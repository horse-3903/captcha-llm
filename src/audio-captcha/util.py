import os
import re
import csv
import base64
import time
import asyncio
import logging
import random
import pandas as pd
import numpy as np
from io import BytesIO
from dotenv import load_dotenv
from scipy.io import wavfile
from openai import AsyncOpenAI
from log import ColourFormatter

load_dotenv()
client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])


# ==========================================================
# Logging Configuration
# ==========================================================
def get_logger(name: str = "AudioCaptchaTester", level=logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColourFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


logger = get_logger()


# ==========================================================
# Audio Utilities
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


def audio_to_base64_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ==========================================================
# Query OpenRouter
# ==========================================================
async def query_openrouter(model: str, prompt: str, audio_base64: str):
    content = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_audio", "input_audio": {"data": audio_base64, "format": "wav"}}
            ]
        }
    ]
    try:
        resp = await client.responses.create(model=model, input=content, max_output_tokens=256)
        return resp.output_text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"
    

def preview_audio_modes(
    base_audio: str,
    background_audio: str = None,
    output_dir: str = "samples",
    gaussian_level: float = 0.7,
    overlap_count: int = 2,
    overlap_ratios: list[float] = None,
    background_boost: float = 1.0,   # <--- NEW
):
    """
    Generate and save sample audio files for all noise types with user-controllable parameters.
    background_boost: controls how loud background noise is (1.0 = same volume, 1.5 = louder).
    """
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"🎧 Generating audio previews from {base_audio}")

    # --- 1. Clean ---
    clean_path = os.path.join(output_dir, "clean.wav")
    with open(base_audio, "rb") as src, open(clean_path, "wb") as dst:
        dst.write(src.read())

    # --- 2. Gaussian noise ---
    gaussian_bytes = add_gaussian_noise(base_audio, noise_level=gaussian_level)
    with open(os.path.join(output_dir, f"gaussian_{gaussian_level:.2f}.wav"), "wb") as f:
        f.write(gaussian_bytes)

    # --- 3. Background noise ---
    if background_audio:
        background_bytes = add_background_noise(base_audio, background_audio, boost=background_boost)
        with open(os.path.join(output_dir, f"background_boost_{background_boost:.1f}.wav"), "wb") as f:
            f.write(background_bytes)
    else:
        logger.info("Skipping background noise: no background_audio provided")

    # --- 4. Combined overlapping noise ---
    all_files = [
        os.path.join(os.path.dirname(base_audio), f)
        for f in os.listdir(os.path.dirname(base_audio))
        if f.endswith(".wav") and f != os.path.basename(base_audio)
    ]

    if len(all_files) >= overlap_count:
        random_others = random.sample(all_files, overlap_count)
        if overlap_ratios is None:
            overlap_ratios = [round(random.uniform(0.3, 0.7), 2) for _ in range(overlap_count)]

        logger.info(f"Combining {len(random_others)} others with ratios={overlap_ratios}")
        combined_bytes = combine_audio_files(base_audio, random_others, ratios=overlap_ratios)
        with open(os.path.join(output_dir, f"combined_{'_'.join(map(str, overlap_ratios))}.wav"), "wb") as f:
            f.write(combined_bytes)
    else:
        logger.warning(f"Not enough files to combine ({len(all_files)} found, need {overlap_count}).")

    logger.info(f"✅ All preview samples saved to: {output_dir}")


# ==========================================================
# Audio CAPTCHA Tester
# ==========================================================
class AudioCaptchaTester:
    def __init__(
        self,
        csv_path: str,
        models: list[str],
        results_dir: str,
        mode="none",
        background_file=None,
        gaussian_level=0.1,
        background_boost=1.0,
        combined_ratios=None,
        concurrency_limit=5,
        sample_limit=None,
    ):
        self.csv_path = csv_path
        self.models = models
        self.results_dir = results_dir
        self.mode = mode
        self.background_file = background_file
        self.gaussian_level = gaussian_level
        self.background_boost = background_boost
        self.combined_ratios = combined_ratios or [0.4, 0.3]
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.sample_limit = sample_limit

        self.logger = get_logger()
        os.makedirs(os.path.join(self.results_dir, "raw"), exist_ok=True)

        self.total = 0
        self.in_flight = {m: 0 for m in models}
        self.completed = {m: 0 for m in models}
        self.timings = {m: [] for m in models}
        self._stop_reporter = asyncio.Event()

    def reset(self):
        self.total = 0
        self.in_flight = {m: 0 for m in self.models}
        self.completed = {m: 0 for m in self.models}
        self.timings = {m: [] for m in self.models}
        self._stop_reporter = asyncio.Event()

    def set_results_dir(self, results_dir):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "raw"), exist_ok=True)
    
    # ---------------------------------------------
    async def _report_progress(self, interval=5):
        start = time.time()
        while not self._stop_reporter.is_set():
            total_done = sum(self.completed.values())
            total_inflight = sum(self.in_flight.values())
            total_tasks = self.total * len(self.models)
            lines = []

            for m in self.models:
                avg_t = np.mean(self.timings[m]) if self.timings[m] else 0
                remain = max(self.total - self.completed[m], 0)
                eta = avg_t * remain if avg_t > 0 else 0
                lines.append(
                    f"{m:30} | in-flight={self.in_flight[m]:2d} | done={self.completed[m]:3d}/{self.total} | ETA={eta:6.1f}s"
                )

            overall = total_done / max(total_tasks, 1)
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
            self.logger.info(
                f"📊 Progress {overall*100:5.1f}% | inflight={total_inflight:2d} | elapsed={elapsed}\n" +
                "\n".join(lines)
            )
            try:
                await asyncio.wait_for(self._stop_reporter.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    # ---------------------------------------------
    async def _transform_audio(self, filename: str, df: pd.DataFrame):
        if self.mode == "none":
            with open(filename, "rb") as f:
                return f.read()
        elif self.mode == "gaussian":
            return add_gaussian_noise(filename, noise_level=self.gaussian_level)
        elif self.mode == "background":
            if not self.background_file:
                raise ValueError("Background file required for background mode.")
            return add_background_noise(filename, self.background_file, boost=self.background_boost)
        elif self.mode == "combined":
            others = df[df["filename"] != filename]["filename"]
            if len(others) < 2:
                raise ValueError("Not enough other samples for combined mode.")
            sample = others.sample(2).tolist()
            return combine_audio_files(filename, sample, ratios=self.combined_ratios)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    # ---------------------------------------------
    async def _process_one(self, model: str, row, df):
        async with self.semaphore:
            self.in_flight[model] += 1
            start_t = time.time()
            try:
                filename = row["filename"]
                question = row["question"]
                answers = [row["a"], row["b"], row["c"], row["d"], row["e"]]
                label = row["answer"]

                audio_bytes = await self._transform_audio(filename, df)
                audio_b64 = audio_to_base64_bytes(audio_bytes)

                prompt = (
                    f"Answer choices:\n"
                    f"A: {answers[0]}\n"
                    f"B: {answers[1]}\n"
                    f"C: {answers[2]}\n"
                    f"D: {answers[3]}\n"
                    f"E: {answers[4]}\n"
                    "Which answer is correct? Respond only with the correct letter with no form of explanation."
                )

                response = await query_openrouter(model, prompt, audio_b64)
                self._save_result(row["id"], model, response, label)
                self.completed[model] += 1
            except Exception as e:
                self.logger.error(f"[{model}] ❌ Error processing row: {e}")
            finally:
                self.in_flight[model] = max(0, self.in_flight[model] - 1)
                self.timings[model].append(time.time() - start_t)

    # ---------------------------------------------
    async def _run_all(self):
        df = pd.read_csv(self.csv_path)
        
        if self.sample_limit:
            df = df.head(self.sample_limit)
        self.total = len(df)

        self.logger.info(f"🎧 Mode={self.mode}, Samples={self.total}, Models={len(self.models)}")
        reporter = asyncio.create_task(self._report_progress(interval=5))

        tasks = []
        for _, row in df.iterrows():
            for model in self.models:
                tasks.append(asyncio.create_task(self._process_one(model, row, df)))

        await asyncio.gather(*tasks)
        self._stop_reporter.set()
        await reporter
        self.logger.info("✅ All processing complete.")

    # ---------------------------------------------
    def _save_result(self, row_id: str, model: str, predicted: str, label: str):
        model_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model).strip('-')
        out_path = os.path.join(self.results_dir, "raw", f"{model_name}-results.csv")
        exists = os.path.isfile(out_path)

        with open(out_path, "a" if exists else "w+", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(["ID", "Predicted", "Correct"])
            writer.writerow([row_id, predicted, label])

    # ---------------------------------------------
    def start(self):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self._run_all())
