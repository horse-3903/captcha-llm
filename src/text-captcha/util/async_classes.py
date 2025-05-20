import os
import csv

import random

import io
import base64
from PIL import Image

from dotenv import load_dotenv

import asyncio
import aiofiles
from tqdm.asyncio import tqdm

from google import genai
from openai import OpenAI, AsyncOpenAI

load_dotenv()

class AsyncCaptchaTester:
    """
    A parent class for testing CAPTCHA-solving models.
    """

    def __init__(self, result_path: str = None, model_lst: list[str] = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        """
        Initialize the AsyncCaptchaTester with configuration parameters.
        """
        self.result_path = result_path
        self.data_path = data_path
        self.model_lst = model_lst
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.delay = delay
        
        os.makedirs(result_path, exist_ok=True)

    def set_model_lst(self, model_lst: list):
        self.model_lst = model_lst
        
    def set_data_path(self, data_path: str):
        self.data_path = data_path
        
    def set_result_path(self, result_path: str):
        self.result_path = result_path
        os.makedirs(result_path, exist_ok=True)

    async def load_prompt(self, prompt_id: int) -> str:
        prompt_path = f"prompts/prompt_{prompt_id}.txt"
        async with aiofiles.open(prompt_path, "r") as f:
            return await f.read()

    def extract_actual_solution(self, image_file_name: str) -> str:
        return image_file_name.split("/")[-1].split(".")[0]
    
    async def process_captcha(self, model: str, prompt: str, image_file_name: str):
        actual_solution = self.extract_actual_solution(image_file_name)
        image_path = f"{self.data_path}/{image_file_name}"
        image = Image.open(image_path)
        delay = self.delay

        for attempt in range(self.max_retries):
            try:
                predicted_solution = await self.run_llm(model, prompt, image)
                predicted_solution = self.sanitise_response(predicted_solution)
                await self.save_result(model, actual_solution, predicted_solution)
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"Model {model} encountered error. Resolving...")
                    await asyncio.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    print(f"Max retries reached for {image_file_name}. Error: {str(e)}")

    async def test_captcha(self, prompt_id: int, no_samples: int, batch_size: int = 10):
        assert self.model_lst, "Model list is empty."
        assert os.path.exists(self.data_path), "Data path does not exist."
        assert os.path.exists(self.result_path), "Results path does not exist."

        prompt = await self.load_prompt(prompt_id)
        image_file_lst = random.sample(os.listdir(self.data_path), no_samples)

        # Create a semaphore to limit concurrent tasks
        semaphore = asyncio.Semaphore(batch_size)

        async def limited_task(model, prompt, image_file_name):
            async with semaphore:
                return await self.process_captcha(model, prompt, image_file_name)

        tasks = []
        for model in self.model_lst:
            for image_file_name in image_file_lst:
                task = limited_task(model, prompt, image_file_name)
                tasks.append(task)
        
        await tqdm.gather(*tasks, desc="Processing CAPTCHAs")
                            
    async def crosscheck_captcha(self, prompt_id: int, no_samples: int, batch_size: int = 10):
        assert self.model_lst, "Model list is empty."
        assert os.path.exists(self.data_path), "Data path does not exist."
        assert os.path.exists(self.result_path), "Results path does not exist."

        prompt = await self.load_prompt(prompt_id)
        data_lst = []
        
        # Read data files
        for data_file in os.listdir(self.data_path):
            data_file_name = f"{self.data_path}/{data_file}"
            async with aiofiles.open(data_file_name, mode='r', encoding="utf-8") as f:
                reader = csv.reader(await f.readlines())
                headers = next(reader)
                if len(headers) == 2:
                    data_lst += [*reader]
        
        # Filter and sample data
        data_lst = [(a, p) for a, p in data_lst if os.path.exists(f"./data/{a}.png")]
        data_lst = random.sample(data_lst, no_samples)
        
        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(batch_size)

        async def process_with_limit(model, prompt, image_file_name):
            async with semaphore:
                return await self.process_captcha(model, prompt, image_file_name)

        tasks = []
        for model in self.model_lst:
            for actual_solution, predicted_solution in tqdm(data_lst, desc=f"Crosschecking {model}"):
                image_file_name = f"{actual_solution}.png"
                
                if not os.path.exists(os.path.join("data", image_file_name)):
                    continue
                
                modified_prompt = prompt + "\n"
                modified_prompt += f"The solution provided earlier was '{predicted_solution}', but I need you to carefully verify this by analyzing the image again.\n"
                modified_prompt += "Ensure that the text matches exactly, including case sensitivity and any possible distortions. If the previous solution is incorrect, provide the corrected text based solely on the image."
                
                tasks.append(process_with_limit(model, modified_prompt, image_file_name))
        
        await tqdm.gather(*tasks, desc="Cross-checking CAPTCHAs")

    async def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def sanitise_response(self, response: str) -> str:
        response = response.replace(",", "").replace("\n", "")
        response = response.encode().decode('unicode_escape')
        return response

    async def save_result(self, model: str, actual_solution: str, predicted_solution: str):
        result_path = self.result_path.rstrip("/")
        os.makedirs(f"{result_path}/raw/", exist_ok=True)
        results_file_name = f"{result_path}/raw/{model.replace("/", "_")}_results.csv"

        # Create file with header if it doesn't exist
        if not os.path.exists(results_file_name):
            async with aiofiles.open(results_file_name, "w") as f:
                await f.write("Actual Solution,Predicted Solution\n")

        # Append the result
        async with aiofiles.open(results_file_name, "a") as f:
            await f.write(f"{actual_solution},{predicted_solution}\n")


class GeminiAsyncCaptchaTester(AsyncCaptchaTester):
    def __init__(self, model_lst: list[str] = None, result_path: str = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        super().__init__(result_path, model_lst, data_path, max_retries, backoff_factor, delay)
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    async def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        def sync_generate():
            return self.client.models.generate_content(
                model=model,
                contents=[prompt, image]
            )
        
        response = await asyncio.to_thread(sync_generate)
        return response.text.strip()

class OpenAIAsyncCaptchaTester(AsyncCaptchaTester):
    def __init__(self, model_lst: list[str] = None, result_path: str = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        super().__init__(result_path, model_lst, data_path, max_retries, backoff_factor, delay)
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
    async def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response = await self.client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()

class OpenRouterAsyncCaptchaTester(AsyncCaptchaTester):
    def __init__(self, model_lst: list[str] = None, result_path: str = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        super().__init__(result_path, model_lst, data_path, max_retries, backoff_factor, delay)
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )
        
    async def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        def sync_generate():
            return self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                        ],
                    }
                ],
                max_tokens=300,
            )
        
        response = await asyncio.to_thread(sync_generate)
        
        if hasattr(response, "error"):
            raise Exception(response.error.get("message"))
            
        return response.choices[0].message.content.strip()