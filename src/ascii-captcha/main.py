import os
import random
from dotenv import load_dotenv
from util import GeminiASCIICaptchaTester

load_dotenv()
API_KEY = os.environ["GEMINI_API_KEY"]

# gemini_model_lst = ["gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
# gemini_model_lst = ["gemini-1.5-flash-8b", 
gemini_model_lst = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
random.shuffle(gemini_model_lst)

# Instantiate tester
tester = GeminiASCIICaptchaTester(
    model_lst=gemini_model_lst,  # Adjust based on your Gemini model name
    result_path="results/ascii-captcha/text",
    data_path="data/ascii-captcha",
    gemini_api_key=API_KEY,
    render_ascii_as_image=False  # Set True if you want to test with rendered images
)

tester.test_ascii_captchas(n=200)