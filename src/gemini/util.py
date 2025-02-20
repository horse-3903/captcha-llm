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

def test_captcha(image_file_name: str, model: str = "gemini-2.0-flash-exp") -> tuple:
    actual_solution = image_file_name.split(".")[0].split("_")[1]
    image = PIL.Image.open(image_file_name)

    max_retries = 8
    backoff_factor = 2
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    "Analyse this image and extract only the CAPTCHA text. The CAPTCHA text is case-sensitive. Keep in mind that there are no whitespaces and there are only alphanumeric characters in the CAPTCHA text. Do not include any additional explanations, descriptions, or formatting—just return the exact CAPTCHA solution as plain text.",
                    image,
                ]
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

def save_result(parent_dir: str, model: str, actual_solution: str, predicted_solution: str):
    parent_dir = parent_dir.rstrip("/")
    results_file_name = f"{parent_dir}/{model}_results.csv"
    
    if not os.path.exists(results_file_name):
        os.makedirs(parent_dir, exist_ok=True)
        
        with open(results_file_name, "w+", encoding="utf-8") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["Actual Solution", "Predicted Solution", "Result"])

    with open(results_file_name, "a", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([actual_solution, predicted_solution, int(actual_solution == predicted_solution)])