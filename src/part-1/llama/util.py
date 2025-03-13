import os
import csv
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
API_KEY = os.environ.get("OPENROUTER_API_KEY")

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=API_KEY,
)

# Function to encode the image in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def test_captcha(prompt: int, image_file_name: str, model: str = "gpt-4o-mini") -> tuple:
    with open(f"prompts/prompt_{prompt}.txt", "r") as f:
        prompt = f.read()
    
    actual_solution = image_file_name.split(".")[0].split("_")[1]
    base64_image = encode_image(image_file_name)

    max_retries = 8
    backoff_factor = 2
    delay = 1

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
            )
            predicted_solution = response.choices[0].message.content.strip()
            return actual_solution, predicted_solution

        except Exception as e:
            if hasattr(response, "error"):
                if attempt < max_retries - 1:
                    print(f"\nRate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    delay *= backoff_factor
                else:
                    print("\nMax retries reached. Skipping this request.")
                    return actual_solution, "ERROR_RATE_LIMIT"
            else:
                raise e

def save_result(parent_dir: str, model: str, actual_solution: str, predicted_solution: str):
    parent_dir = parent_dir.rstrip("/")
    results_file_name = f"{parent_dir}/{model.split('/')[1].replace(':free', '')}_results.csv"
    
    os.makedirs(parent_dir, exist_ok=True)
    
    file_exists = os.path.exists(results_file_name)

    with open(results_file_name, "a", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        if not file_exists:
            csv_writer.writerow(["Actual Solution", "Predicted Solution", "Result"])
        csv_writer.writerow([actual_solution, predicted_solution.replace(",", "-"), int(actual_solution == predicted_solution)])