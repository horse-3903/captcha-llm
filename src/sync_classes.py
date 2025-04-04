import os
import csv
import time
import random
import io
import base64
from PIL import Image
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

class CaptchaTester:
    """
    A parent class for testing CAPTCHA-solving models.
    """

    def __init__(self, result_path: str = None, model_lst: list[str] = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        """
        Initialize the CaptchaTester with configuration parameters.

        Args:
            result_path (str): Directory to save results.
            max_retries (int): Maximum number of retries for failed requests.
            backoff_factor (int): Factor by which the delay increases after each retry.
            delay (int): Initial delay between retries in seconds.
        """
        self.result_path = result_path
        self.data_path = data_path
        self.model_lst = model_lst
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.delay = delay

    def set_model_lst(self, model_lst: list):
        """
        Set the list of models to be tested.

        Args:
            model_lst (list): List of model names.
        """
        self.model_lst = model_lst
        
    def set_data_path(self, data_path: str):
        """
        Set the path to the data files.

        Args:
            data_path (str): Path to data.
        """
        self.data_path = data_path
        
    def set_result_path(self, result_path: str):
        """
        Set the path to the result files.

        Args:
            result_path (str): Path to result.
        """
        self.result_path = result_path

    def load_prompt(self, prompt_id: int) -> str:
        """
        Load a prompt from a file based on the prompt ID.

        Args:
            prompt_id (int): ID of the prompt to load.

        Returns:
            str: The content of the prompt file.
        """
        prompt_path = f"prompts/prompt_{prompt_id}.txt"
        
        with open(prompt_path, "r") as f:
            return f.read()

    def extract_actual_solution(self, image_file_name: str) -> str:
        """
        Extract the actual solution from the image file name.

        Args:
            image_file_name (str): Name of the image file.

        Returns:
            str: The actual solution extracted from the file name.
        """
        return image_file_name.split(".")[0]
    
    def process_captcha(self, model: str, prompt: str, image_file_name: str):
        """
        Process CAPTCHA with model and prompt

        Args:
            model (str): _description_
            prompt (str): _description_
            image_file_name (str): _description_
        """
        actual_solution = self.extract_actual_solution(image_file_name)
        image_path = f"{self.data_path}/{image_file_name}"
        image = Image.open(image_path)
        delay = self.delay

        for attempt in range(self.max_retries):
            try:
                predicted_solution = self.run_llm(model, prompt, image)
                predicted_solution = self.sanitise_response(predicted_solution)
                self.save_result(model, actual_solution, predicted_solution)
                break
            except Exception as e:
                if attempt < self.max_retries - 1:
                    time.sleep(delay)
                    delay *= self.backoff_factor
                else:
                    print(f"Max retries reached for {image_file_name}. Skipping.")

    def test_captcha(self, prompt_id: int, no_samples: int):
        """
        Test CAPTCHA-solving models using a specific prompt.

        Args:
            prompt_id (int): ID of the prompt to use for testing.
            no_samples (int): Number of samples to take from.
        """
        assert self.model_lst, "Model list is empty."
        assert os.path.exists(self.data_path), "Data path does not exist."
        assert os.path.exists(self.result_path), "Results path does not exist."

        prompt = self.load_prompt(prompt_id)
        image_file_lst = random.sample(os.listdir(self.data_path), no_samples)

        for model in self.model_lst:
            for image_file_name in tqdm(image_file_lst, desc=f"Processing {model}"):
                self.process_captcha(model, prompt, image_file_name)
                            
    def crosscheck_captcha(self, prompt_id: int, no_samples: int):
        """
        Crosscheck CAPTCHA-solving models using a specific prompt.

        Args:
            prompt_id (int): ID of the prompt to use for testing.
            no_samples (int): Number of samples to take from.
        """
        assert self.model_lst, "Model list is empty. Set models using set_model_lst()."
        assert os.path.exists(self.data_path), "Data path does not exist. Use a valid path for data."
        assert os.path.exists(self.result_path), "Results path does not exist. Use a valid path for data."

        prompt = self.load_prompt(prompt_id)
        
        data_lst = []
        
        for data_file in os.listdir(self.data_path):
            data_file_name = f"{self.data_path}/{data_file}"
            with open(data_file_name) as f:
                reader = csv.reader(f)
                headers = next(reader)
                if len(headers) == 2:
                    data_lst += [*reader]
        
        data_lst = [(actual_solution, predicted_solution) for actual_solution, predicted_solution in data_lst if os.path.exists(f"./data/{actual_solution}.png")]
        
        assert data_lst, "Data path did not have the correct data type."
    
        data_lst = random.sample(data_lst, no_samples)
        
        for model in self.model_lst:
            for actual_solution, predicted_solution in tqdm(data_lst, desc=f"Crosschecking {model}"):
                image_file_name = f"./data/{actual_solution}.png"
                
                if not os.path.exists(image_file_name):
                    continue
                
                modified_prompt = prompt + "\n"
                modified_prompt += f"The solution provided earlier was '{predicted_solution}', but I need you to carefully verify this by analyzing the image again.\n"
                modified_prompt += "Ensure that the text matches exactly, including case sensitivity and any possible distortions. If the previous solution is incorrect, provide the corrected text based solely on the image."
                
                self.process_captcha(model, modified_prompt, image_file_name)

    def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        """
        Run the language model to solve the CAPTCHA.

        Args:
            model (str): Name of the model.
            prompt (str): Prompt to provide to the model.
            image (PIL.Image): CAPTCHA image.

        Returns:
            str: Predicted solution.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def sanitise_response(self, response: str) -> str:
        """
        Sanitise the model's response by removing unwanted characters.

        Args:
            response (str): Raw response from the model.

        Returns:
            str: Sanitised response.
        """
        return response.replace(",", "").replace("\n", "")

    def save_result(self, model: str, actual_solution: str, predicted_solution: str):
        """
        Save the test results to a CSV file.

        Args:
            model (str): Name of the model.
            actual_solution (str): Actual solution of the CAPTCHA.
            predicted_solution (str): Predicted solution by the model.
        """
        result_path = self.result_path.rstrip("/")
        results_file_name = f"{result_path}/raw/{model}_results.csv"

        # Create the results directory and file if they don't exist
        if not os.path.exists(results_file_name):
            os.makedirs(f"{result_path}/raw/", exist_ok=True)
            
            header = ["Actual Solution", "Predicted Solution"]

            with open(results_file_name, "w+", encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(header)

        # Append the result to the CSV file
        with open(results_file_name, "a", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            
            result = [actual_solution, predicted_solution]
            
            csv_writer.writerow(result)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

class GeminiCaptchaTester(CaptchaTester):
    def __init__(self, model_lst: list[str] = None, result_path: str = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        super().__init__(result_path, model_lst, data_path, max_retries, backoff_factor, delay)
    
    def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        response = gemini_client.models.generate_content(
            model=model,
            contents=[prompt, image]
        )
        
        predicted_solution = response.text
        
        return predicted_solution.strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

class OpenAICaptchaTester(CaptchaTester):
    def __init__(self, model_lst: list[str] = None, result_path: str = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        super().__init__(result_path, model_lst, data_path, max_retries, backoff_factor, delay)
        
    def encode_image(self, image: Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded_image
    
    def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        base64_image = self.encode_image(image)
        
        response = openai_client.chat.completions.create(
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
        )
        predicted_solution = response.choices[0].message.content
        
        return predicted_solution.strip()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

class OpenRouterCaptchaTester(CaptchaTester):
    def __init__(self, model_lst: list[str] = None, result_path: str = None, data_path: str = None, max_retries: int = 8, backoff_factor: int = 2, delay: int = 1):
        super().__init__(result_path, model_lst, data_path, max_retries, backoff_factor, delay)
        
    def encode_image(self, image: Image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return encoded_image
    
    def run_llm(self, model: str, prompt: str, image: Image.Image) -> str:
        base64_image = self.encode_image(image)
        
        response = openrouter_client.chat.completions.create(
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
        )
        predicted_solution = response.choices[0].message.content
        
        return predicted_solution.strip()