import time
import os
import csv
from dotenv import load_dotenv
from google import genai
import PIL.Image
from google.genai.errors import ClientError

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def test_captcha(prompt: int, image_file_name: str, model: str = "gemini-2.0-flash-exp") -> tuple:
    with open(f"prompts/prompt_{prompt}.txt", "r") as f:
        prompt = f.read()
        
    actual_solution = image_file_name.split(".")[0].split("_")[1]
    image = PIL.Image.open(image_file_name)

    max_retries = 8
    backoff_factor = 2
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, image]
            )
            predicted_solution = response.text
            
            return actual_solution, predicted_solution

        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    print(f"\nRate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    print("\nMax retries reached. Skipping this request.")
                    print()
                    return actual_solution, "ERROR_RATE_LIMIT"
            else:
                raise e
            
def crosscheck_captcha(prompt: int, actual_solution: str, pred_solution: str, model: str = "gemini-2.0-flash-exp") -> tuple:
    with open(f"prompts/prompt_{prompt}.txt", "r") as f:
        prompt = f.read()
        prompt += "\nThe solution provided earlier was '{pred_solution}', but I need you to carefully verify this by analyzing the image again. Ensure that the text matches exactly, including case sensitivity and any possible distortions. If the previous solution is incorrect, provide the corrected text based solely on the image."
        
    image = PIL.Image.open(f"./data/{actual_solution}.png")

    max_retries = 8
    backoff_factor = 2
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[prompt, image]
            )
            cross_pred_solution = response.text
            
            return actual_solution, pred_solution, cross_pred_solution.replace(",", "")

        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                if attempt < max_retries - 1:
                    print(f"\nRate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    print("\nMax retries reached. Skipping this request.")
                    print()
            else:
                raise e
    

def save_result(parent_dir: str, results_file_name: str, data: list[tuple[str, str, str]]):   
    parent_dir = parent_dir.rstrip("/") + "/"
    
    if not os.path.exists(parent_dir + results_file_name):
        os.makedirs(parent_dir, exist_ok=True)
        
        with open(parent_dir + results_file_name, "w+", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Actual Solution", "Predicted Solution", "Cross Checked Solution"])

    with open(parent_dir + results_file_name, "a", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)