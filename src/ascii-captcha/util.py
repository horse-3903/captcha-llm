import os
import time
import csv
import random
from typing import List
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from tqdm import tqdm
from google.genai import Client as GeminiClient

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def text_to_image(text: str, font_path: str = os.path.join(root_path, "fonts", "courier.ttf"), font_size: int = 16) -> Image.Image:
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

    return image

class GeminiASCIICaptchaTester:
    """
    A tester class for solving ASCII CAPTCHAs using Google's Gemini API.
    Handles both raw ASCII text and optional image rendering for model input.
    """

    def __init__(
        self,
        model_lst: List[str],
        result_path: str,
        data_path: str,
        gemini_api_key: str,
        max_retries: int = 8,
        backoff_factor: int = 2,
        delay: int = 1,
        render_ascii_as_image: bool = False
    ):
        self.model_lst = model_lst
        self.result_path = os.path.join(root_path, result_path)
        self.data_path = os.path.join(root_path, data_path)
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.delay = delay
        self.render_ascii_as_image = render_ascii_as_image
        self.gemini_client = GeminiClient(api_key=gemini_api_key)

        os.makedirs(os.path.join(result_path, "raw"), exist_ok=True)

    def read_ascii_samples(self, n: int) -> List[str]:
        files = os.listdir(self.data_path)
        samples = random.sample(files, n)
        return [os.path.join(self.data_path, f) for f in samples]

    def process_sample(self, model: str, ascii_text: str, label: str):
        prompt = f"{ascii_text}\n\nRespond with only the CAPTCHA solution. Do not include any explanation, formatting, or extra characters." if not self.render_ascii_as_image else "Solve the CAPTCHA from the image. Respond with only the CAPTCHA solution. Do not include any explanation, formatting, or extra characters."
        content = [prompt]
        if self.render_ascii_as_image:
            image = text_to_image(ascii_text)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            content.append(buffer.getvalue())  # type: ignore

        attempt = 0
        while attempt < self.max_retries:
            try:
                predicted_solution = self.run_llm(model, content)
                self.save_result(model, label, predicted_solution)
                return
            except Exception as e:
                print(f"[Attempt {attempt + 1}] Error: {e}")
                time.sleep(self.delay * (self.backoff_factor ** attempt))
                attempt += 1

        self.save_result(model, label, "[FAILED]")

    def run_llm(self, model: str, content: list) -> str:
        if len(content) == 1:
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=[content[0]]
            )
        else:
            image = Image.open(BytesIO(content[1]))
            response = self.gemini_client.models.generate_content(
                model=model,
                contents=[content[0], image]
            )

        return response.text.strip() if response.text else "[NO RESPONSE]"

    def test_ascii_captchas(self, n: int):
        sample_paths = self.read_ascii_samples(n)
        for model in self.model_lst:
            for path in tqdm(sample_paths, desc=f"Processing {model}"):
                with open(path, "r") as f:
                    ascii_text = f.read()
                    label = os.path.splitext(os.path.basename(path))[0]
                    self.process_sample(model, ascii_text, label)

    def save_result(self, model: str, actual_solution: str, predicted_solution: str):
        results_file_name = os.path.join(self.result_path, "raw", f"{model}_results.csv")
        file_exists = os.path.isfile(results_file_name)

        with open(results_file_name, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Actual Solution", "Predicted Solution"])
            writer.writerow([actual_solution, predicted_solution.strip()])
