import os
import glob

import io

import csv
import random

from PIL import Image, ImageDraw, ImageFont
import base64

import asyncio
from ollama import AsyncClient
import logging

from difflib import SequenceMatcher

import itertools
from collections import defaultdict

from log import ColourFormatter

def setup_logger(level=logging.DEBUG):
    handler = logging.StreamHandler()
    handler.setFormatter(ColourFormatter())

    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger

root_path = os.getcwd()

def pil_to_b64(pil_img, fmt="PNG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def text_to_image(
    text: str, 
    font_path: str = os.path.join(root_path, "fonts", "courier.ttf"), 
    font_size: int = 16
) -> str:

    lines = text.split("\n")
    font = ImageFont.truetype(font_path, font_size)

    max_width = max(font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines)
    total_height = sum(font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines)

    image = Image.new("RGB", (int(max_width + 10), int(total_height + 10)), color="white")
    draw = ImageDraw.Draw(image)

    y = 5
    for line in lines:
        draw.text((5, y), line, fill="black", font=font)
        y += font.getbbox(line)[3] - font.getbbox(line)[1]

    b64_image = pil_to_b64(image)

    return b64_image

class ASCIICaptchaTester:
    def __init__(
        self,
        client: AsyncClient,
        data_path: str = "data/ascii-captcha",
        result_path: str = "results",
    ) -> None:
        self.client = client
        self.result_path = os.path.join(root_path, result_path)
        self.data_path = os.path.join(root_path, data_path)

    async def save_test(self, file_path: str, result_lst: list[tuple[str, str]]) -> None:
        result_cleaned_lst = []

        for value, response in result_lst:
            response = response.replace(',', '')
            result_cleaned_lst.append((
                value, response, int(value == response), 
                similar(value, response), similar(value.lower(), response.lower())
            ))
        
        with open(file_path, "w+", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Actual Solution", "Predicted Solution", "Accuracy", "Cased Similarity", "Uncased Similarity"])
            writer.writerows(result_cleaned_lst)

    async def run_test_suite(
        self,
        name: str,
        message_lst: list[dict[str, str]], 
        model_lst: list[str],
        num_samples: int = None,
        max_concurrency: int = 6
    ) -> None:
        os.makedirs(os.path.join(self.result_path, name), exist_ok=True)
        await self.run_test(name, message_lst, model_lst, render_image=False, num_samples=num_samples, max_concurrency=max_concurrency)
        await self.run_test(name, message_lst, model_lst, render_image=True, num_samples=num_samples, max_concurrency=max_concurrency)
    
    async def run_test(
        self, 
        name: str,
        message_lst: list[dict[str, str]], 
        model_lst: list[str],
        num_samples: int = None,
        render_image: bool = False,
        max_concurrency: int = 6
    ) -> None:
        logger = setup_logger(logging.DEBUG)

        logger.info("Fetching available models from Ollama Client...")
        available_models = [m.model for m in (await self.client.list()).models]
        logger.info(f"Available models: {available_models}")
        assert all(model in available_models for model in model_lst)

        logger.info(f"Collecting samples from {self.data_path}...")
        sample_lst = glob.glob(os.path.join(self.data_path, "*.txt"))
        logger.info(f"Found {len(sample_lst)} samples.")

        if num_samples is None:
            num_samples = len(sample_lst)
        sample_lst = random.sample(sample_lst, num_samples)
        logger.info(f"Randomly selected {num_samples} samples.")

        base_dir = os.path.join(self.result_path, name, "image" if render_image else "text")
        os.makedirs(base_dir, exist_ok=True)

        sem = asyncio.Semaphore(max_concurrency)

        in_flight = 0
        completed = 0
        failed = 0

        async def meter():
            while True:
                logger.debug(f"[meter] in_flight={in_flight} completed={completed} failed={failed}")
                await asyncio.sleep(2)

        async def process_one(sample_path: str, model: str) -> tuple[str, str, str]:
            """Return (model, sample_value, response_text)."""
            nonlocal in_flight, completed, failed
            sample_value = os.path.splitext(os.path.basename(sample_path))[0]

            if render_image:
                sample_text = await asyncio.to_thread(lambda: open(sample_path, "r").read())
                sample_data = await asyncio.to_thread(text_to_image, sample_text)
                sample_prompt_msg = {"role": "user", "images": [sample_data]}
            else:
                sample_data = await asyncio.to_thread(lambda: open(sample_path, "r").read())
                sample_prompt_msg = {"role": "user", "content": sample_data}

            message_lst_prompt = [*message_lst, sample_prompt_msg]

            async with sem:
                in_flight += 1
                try:
                    resp = (await asyncio.wait_for(
                        self.client.chat(model=model, messages=message_lst_prompt),
                        timeout=120
                    )).message.content
                    completed += 1
                    return model, sample_value, str(resp)
                except Exception as e:
                    failed += 1
                    logger.warning(f"[{model}] Sample {sample_value} failed: {e}")
                    return model, sample_value, ""
                finally:
                    in_flight -= 1

        all_pairs = list(itertools.product(sample_lst, model_lst))
        logger.info(f"Queuing {len(all_pairs)} total requests across {len(model_lst)} models with max_concurrency={max_concurrency}.")

        tasks = [asyncio.create_task(process_one(s, m)) for (s, m) in all_pairs]

        meter_task = asyncio.create_task(meter())
        results = await asyncio.gather(*tasks)
        meter_task.cancel()
        try:
            await meter_task
        except asyncio.CancelledError:
            pass

        by_model = defaultdict(list)
        for model, sample_value, resp_text in results:
            by_model[model].append((sample_value, resp_text))

        for model, pairs in by_model.items():
            pairs.sort(key=lambda x: x[0])
            file_path = os.path.join(base_dir, f"{model.replace(':', '-')}.csv")
            logger.info(f"Saving results for {model} → {file_path}")
            await self.save_test(file_path, pairs)

        logger.info("Testing complete.")
