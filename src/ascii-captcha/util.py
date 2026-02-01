import os
import csv
import glob
import base64
import random
import asyncio
import logging
import re
import time
import requests
from io import BytesIO
from typing import List
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from dotenv import load_dotenv

# --- Provider adapters ---
from utils.retry import with_retries
from utils.rate_limiter import RateLimiter
from utils.openai import query_openai
# from utils.anthropic import query_anthropic
# from utils.gemini import query_gemini
from utils.openrouter import query_openrouter

from log import ColourFormatter


# --------------------------
# Logging Configuration
# --------------------------
def get_logger(name: str = "ASCIICaptchaTester", level: int = logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(ColourFormatter())
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


# --------------------------
# Environment + Paths
# --------------------------
load_dotenv()
ROOT_PATH = os.getcwd()


# --------------------------
# Utility: ASCII → Image
# --------------------------
def text_to_image(
    text: str,
    font_path: str = os.path.join(ROOT_PATH, "fonts", "courier.ttf"),
    font_size: int = 16,
) -> Image.Image:
    """Render ASCII text as a monospaced image."""
    lines = text.splitlines()
    font = ImageFont.truetype(font_path, font_size)
    line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
    total_height = int(sum(line_heights) + 10)
    max_width = int(max((font.getbbox(line)[2] - font.getbbox(line)[0]) for line in lines) + 10)

    image = Image.new("RGB", (max_width, total_height), "white")
    draw = ImageDraw.Draw(image)
    y = 5
    for line, h in zip(lines, line_heights):
        draw.text((5, y), line, fill="black", font=font)
        y += h
    return image


# --------------------------
# Unified ASCII CAPTCHA Tester
# --------------------------
class ASCIICaptchaTester:
    """Unified ASCII CAPTCHA tester supporting OpenAI, Anthropic, and Gemini providers."""

    def __init__(
        self,
        model_list: List[str],
        result_path: str,
        data_path: str,
        concurrency_limit: int = 5,
        rate_limits: dict | None = None,
    ):
        self.model_list = model_list
        self.result_path = os.path.join(ROOT_PATH, result_path)
        self.data_path = os.path.join(ROOT_PATH, data_path)
        self.logger = get_logger()
        self.semaphore = asyncio.Semaphore(concurrency_limit)
        self.active_tasks = 0

        self.in_flight = {}
        self.completed = {}
        self.total = 0
        self.model_timings = {}  # model: list of processing durations

        self._reporter_task = None
        self._stop_reporter = asyncio.Event()

        os.makedirs(os.path.join(self.result_path, "raw"), exist_ok=True)

        rate_limits = rate_limits or {}

        self.rate_limits = {
            provider: RateLimiter(rate=rpm, per=60)
            for provider, rpm in rate_limits.items()
        }

        self.global_rate_limit = rate_limits.get("global", None)

        self.global_limiter = (
            RateLimiter(rate=self.global_rate_limit, per=60)
            if self.global_rate_limit
            else None
        )

        self.disabled_models = set()

    # --------------------------
    # Data
    # --------------------------
    def read_ascii_samples(self, n: int) -> List[str]:
        files = glob.glob(os.path.join(self.data_path, "**", "*.txt"), recursive=True)
        return random.sample(files, min(n, len(files)))

    # --------------------------
    # Utility
    # --------------------------
    def fetch_image_b64(self, url: str) -> str | None:
        try:
            resp = requests.get(url)
            resp.raise_for_status()
            return base64.b64encode(resp.content).decode()
        except Exception as e:
            self.logger.warning(f"⚠️ Could not fetch image: {e}")
            return None

    async def _report_in_flight(self, interval: int = 5):
        """Periodically logs progress, per-model and overall ETA."""
        start_time = time.time()
        while not self._stop_reporter.is_set():
            total_inflight = 0
            total_done = 0
            total_eta = 0
            total_samples = self.total * len(self.model_list)

            lines = []
            now = time.time()

            model_names = sorted(self.in_flight.keys())
            max_name_len = max((len(m) for m in model_names), default=10)

            for model in model_names:
                inflight = self.in_flight.get(model, 0)
                done = self.completed.get(model, 0)
                total_inflight += inflight
                total_done += done

                avg_time = 0
                if self.model_timings.get(model):
                    avg_time = sum(self.model_timings[model]) / len(self.model_timings[model])
                remaining = max(self.total - done, 0)
                eta = avg_time * remaining if avg_time > 0 else 0
                total_eta += eta
                eta_str = time.strftime("%M:%S", time.gmtime(eta)) if eta > 0 else "--:--"

                lines.append(
                    f"{model.ljust(max_name_len)} | in flight: {str(inflight).rjust(3)} "
                    f"| done: {str(done).rjust(4)} / {self.total} | ETA: {eta_str}"
                )

            # --- Compute overall ETA ---
            overall_progress = total_done / total_samples if total_samples > 0 else 0
            avg_per_task = (
                sum(sum(v) for v in self.model_timings.values()) /
                max(sum(len(v) for v in self.model_timings.values()), 1)
            )
            remaining_tasks = total_samples - total_done
            global_eta_sec = remaining_tasks * avg_per_task if avg_per_task > 0 else 0
            global_eta_str = time.strftime("%H:%M:%S", time.gmtime(global_eta_sec)) if global_eta_sec > 0 else "--:--"

            elapsed = time.strftime("%H:%M:%S", time.gmtime(now - start_time))
            summary = "\n".join(lines) if lines else "No active requests"

            self.logger.info(
                f"📊 In-flight summary (total={total_inflight}, elapsed={elapsed})\n"
                f"{summary}\n"
                f"🕒 Global ETA (all models): {global_eta_str} (progress {overall_progress*100:.1f}%)"
            )

            try:
                await asyncio.wait_for(self._stop_reporter.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue

    # --------------------------
    # Unified Query Dispatcher
    # --------------------------
    async def unified_query(self, model: str, prompt: str, img_b64: str | None):
        provider, model_name = model.split("/", 1)
        limiter = self.rate_limits.get(model, self.rate_limits.get(provider))

        async def _dispatch():
            if provider == "openai":
                return await query_openai(model_name, prompt, img_b64)
            # elif provider == "anthropic":
            #     return await query_anthropic(model_name, prompt, img_b64)
            # elif provider == "gemini":
            #     return await query_gemini(model_name, prompt, img_b64)
            # else:
            #     return await query_openrouter(provider, model_name, prompt, img_b64)
            
            return await query_openrouter(provider, model_name, prompt, img_b64)

        try:
            if self.global_limiter:
                await self.global_limiter.acquire()

            if limiter:
                await limiter.acquire()

            async with self.semaphore:
                response = await with_retries(model, _dispatch, retries=2, base_delay=2, logger=self.logger)
                self.logger.debug(f"Response from {model}: {response}")
                return response
            
        except Exception as e:
            self.logger.error(f"[{model}] ❌ Query failed after retries: {e}")

            if "rate" in str(e).lower() or "limit" in str(e).lower():
                if model not in self.disabled_models:
                    self.logger.warning(f"[{model}] 🚫 Disabling model due to repeated rate-limit failures.")
                    self.disabled_models.add(model)
            return None  # ensure caller knows the query failed

    # --------------------------
    # Response Validator
    # --------------------------
    def validate_output(self, text: str) -> str:
        """Detect irregular or missing output."""
        text = text.strip()
        
        if not text or len(text) == 0:
            self.logger.debug("[MISSING]")
        elif len(text) > 25:
            self.logger.debug(f"[TOO LONG]: {text[:20]}...")
        elif re.search(r"[.!?]", text):
            self.logger.debug(f"[SENTENCE]: {text}")
        elif re.match(r"\[ERROR", text):
            self.logger.debug(f"[ERROR]: {text}")
        elif "\n" in text:
            self.logger.debug(f"[NEWLINE DETECTED]: {text}")
        elif "#" in text:
            self.logger.debug(f"[HASH DETECTED]: {text}")
        else:
            return text

    # --------------------------
    # Process One CAPTCHA
    # --------------------------
    async def process_one(
        self, model: str, path: str, prompt: str, render_ascii_as_image: bool
    ):
        """Send one ASCII CAPTCHA to the model."""
        if model in self.disabled_models:
            return False
    
        with open(path, "r", encoding="utf-8") as f:
            ascii_text = f.read().strip()

        combined_prompt = f"{ascii_text}\n\n{prompt}\nOutput only the exact text."

        self.in_flight[model] = self.in_flight.get(model, 0) + 1
        start = time.time()
        try:
            if render_ascii_as_image:
                img = text_to_image(ascii_text)
                buf = BytesIO()
                img.save(buf, format="PNG")
                img_b64 = base64.b64encode(buf.getvalue()).decode()
                raw_response = await self.unified_query(model, combined_prompt, img_b64)
            else:
                raw_response = await self.unified_query(model, combined_prompt, None)
        finally:
            self.in_flight[model] = max(0, self.in_flight.get(model, 1) - 1)

        duration = time.time() - start
        self.model_timings.setdefault(model, []).append(duration)

        if raw_response:
            validated = self.validate_output(raw_response)
            label = os.path.splitext(os.path.basename(path))[0]
            
            if validated:
                self.completed[model] = self.completed.get(model, 0) + 1
                self.save_result_with_time(model, label.lower(), validated, duration)

    # --------------------------
    # Global Runner
    # --------------------------
    async def run_all_models(self, n: int, prompt: str, render_as_image=False, report_interval: int = 10):
        if getattr(self, "_reporter_task", None) is None:
            self._stop_reporter.clear()
            self._reporter_task = asyncio.create_task(self._report_in_flight(interval=report_interval))

        self.total = n
        for model in self.model_list:
            self.completed[model] = 0
            self.in_flight[model] = 0
            self.model_timings[model] = []

        samples = self.read_ascii_samples(n)
        self.logger.info(f"🧩 Starting multi-model test ({len(self.model_list)} models, {n} CAPTCHAs)")

        async def model_runner(model):
            remaining = list(samples)
            attempts = 0
            while remaining:
                failed = []
                self.logger.info(f"[{model}] ▶ Attempt {attempts+1}, remaining: {len(remaining)}")
                for path in remaining:
                    success = await self.process_one(model, path, prompt, render_as_image)
                    if not success:
                        failed.append(path)
                remaining = failed
                attempts += 1
                if remaining:
                    self.logger.warning(f"[{model}] ⚠️ Retrying {len(remaining)} failed samples...")
                    await asyncio.sleep(5)  # small cooldown
            self.logger.info(f"[{model}] ✅ All {n} CAPTCHAs processed successfully.")

        tasks = [asyncio.create_task(model_runner(model)) for model in self.model_list]
        await asyncio.gather(*tasks)

        self._stop_reporter.set()
        if self._reporter_task:
            await self._reporter_task

        self.logger.info("✅ All models completed successfully.")
        if self.disabled_models:
            self.logger.warning(f"🚫 Disabled models during run: {', '.join(self.disabled_models)}")


    # --------------------------
    # Save Results
    # --------------------------
    def save_result(self, model: str, actual: str, predicted: str):
        os.makedirs(os.path.join(self.result_path, "raw"), exist_ok=True)

        model_name = re.sub(r'[^A-Za-z0-9._-]+', '-', model)
        model_name = re.sub(r'-{2,}', '-', model_name).strip('-') or 'model'

        csv_path = os.path.join(self.result_path, "raw", f"{model_name}-results.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Actual Solution", "Predicted Solution"])
            writer.writerow([actual.strip(), predicted.strip()])
        self.logger.debug(f"[{model}] Saved result for {actual}")

    def save_result_with_time(
        self, model: str, actual: str, predicted: str, response_time_sec: float
    ):
        os.makedirs(os.path.join(self.result_path, "raw"), exist_ok=True)

        model_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model)
        model_name = re.sub(r"-{2,}", "-", model_name).strip("-") or "model"

        csv_path = os.path.join(self.result_path, "raw", f"{model_name}-results.csv")
        file_exists = os.path.isfile(csv_path)

        with open(csv_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    ["Actual Solution", "Predicted Solution", "Response Time (s)"]
                )
            writer.writerow(
                [actual.strip(), predicted.strip(), f"{response_time_sec:.4f}"]
            )
        self.logger.debug(f"[{model}] Saved result for {actual} (t={response_time_sec:.4f}s)")

    def write_timings_summary(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Model",
                    "Samples",
                    "Total Time (s)",
                    "Mean (s)",
                    "Median (s)",
                    "P95 (s)",
                    "Min (s)",
                    "Max (s)",
                ]
            )
            for model, timings in self.model_timings.items():
                if not timings:
                    continue
                arr = sorted(timings)
                count = len(arr)
                total = sum(arr)
                mean = total / count
                median = arr[count // 2] if count % 2 == 1 else (arr[count // 2 - 1] + arr[count // 2]) / 2
                p95_index = max(int(round(0.95 * (count - 1))), 0)
                p95 = arr[p95_index]
                writer.writerow(
                    [
                        model,
                        count,
                        f"{total:.4f}",
                        f"{mean:.4f}",
                        f"{median:.4f}",
                        f"{p95:.4f}",
                        f"{arr[0]:.4f}",
                        f"{arr[-1]:.4f}",
                    ]
                )

    # --------------------------
    # Debug Mode
    # --------------------------
    async def debug_test_mode(
        self,
        prompt="Say hello! What is shown in this image?",
        image="https://images.dog.ceo/breeds/kuvasz/n02104029_4456.jpg",
    ):
        """Simple connectivity test for all models."""
        self.logger.info("🧪 Running debug test mode...")
        img_b64 = self.fetch_image_b64(image)
        tasks = [
            asyncio.create_task(self.unified_query(m, prompt, img_b64))
            for m in self.model_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for model, result in zip(self.model_list, results):
            if isinstance(result, Exception) or "error" in str(result).lower():
                self.logger.error(f"[{model}] ❌ {result}")
            else:
                assert isinstance(result, str)
                self.logger.info(f"[{model}] ✅ {result[:180]}{'...' if len(result) > 180 else ''}")
        self.logger.info("🔍 Debug test complete.")
