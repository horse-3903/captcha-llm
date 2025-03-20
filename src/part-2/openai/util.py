import os
import csv
import time
import base64
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# Function to encode the image in base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
        
def crosscheck_captcha(prompt: int, actual_solution: str, pred_solution: str, model: str = "gpt-4o-mini") -> tuple:
            with open(f"prompts/prompt_{prompt}.txt", "r") as f:
                prompt_text = f.read()
                prompt_text += f"\nThe solution provided earlier was '{pred_solution}', but I need you to carefully verify this by analyzing the image again. Ensure that the text matches exactly, including case sensitivity and any possible distortions. If the previous solution is incorrect, provide the corrected text based solely on the image."

            base64_image = encode_image(f"./data/{actual_solution}.png")

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
                                    {"type": "text", "text": prompt_text},
                                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                                ],
                            }
                        ],
                    )
                    cross_pred_solution = response.choices[0].message.content.strip()
                    return actual_solution, pred_solution, cross_pred_solution.replace(",", "-")

                except Exception as e:
                    if "rate limit" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"\nRate limit exceeded. Retrying in {delay} seconds... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            delay *= backoff_factor
                        else:
                            print("\nMax retries reached. Skipping this request.")
                            return actual_solution, pred_solution, "ERROR_RATE_LIMIT"
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