import sys
sys.path.append("./src/")

import asyncio
import random

from util.sync_classes import CaptchaTester, GeminiCaptchaTester, OpenAICaptchaTester, OpenRouterCaptchaTester
from util.async_classes import AsyncCaptchaTester, GeminiAsyncCaptchaTester, OpenAIAsyncCaptchaTester, OpenRouterAsyncCaptchaTester

# gemini_model_lst = ["gemini-1.5-flash-8b", "gemini-1.5-pro", [
gemini_model_lst = ["gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
gemini_tester = GeminiCaptchaTester(model_lst=gemini_model_lst)

for i in range(3):
    prompt_id = i+1
    gemini_tester.set_data_path(f"./results/text-captcha/test-captcha/prompt-{prompt_id}/gemini/raw")
    gemini_tester.set_result_path(f"./results/text-captcha/crosscheck-captcha/prompt-{prompt_id}/")
    
    gemini_tester.crosscheck_captcha(prompt_id, no_samples=500)