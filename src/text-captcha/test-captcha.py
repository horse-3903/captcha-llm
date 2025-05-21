import sys
sys.path.append("./src/")

import asyncio
import random

from util.sync_classes import CaptchaTester, GeminiCaptchaTester, OpenAICaptchaTester, OpenRouterCaptchaTester
from util.async_classes import AsyncCaptchaTester, GeminiAsyncCaptchaTester, OpenAIAsyncCaptchaTester, OpenRouterAsyncCaptchaTester

gemini_model_lst = ["gemini-1.5-flash-8b", "gemini-1.5-pro", "gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash"]
# gemini_model_lst = ["gemini-1.5-flash-8b", "gemini-1.5-pro"]
openai_model_lst = ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
# openrouter_model_lst_free = ["allenai/molmo-7b-d:free", "qwen/qwen2.5-vl-3b-instruct:free", "qwen/qwen-2.5-vl-7b-instruct:free", "qwen/qwen2.5-vl-32b-instruct:free", "qwen/qwen2.5-vl-72b-instruct:free", "bytedance-research/ui-tars-72b:free", "mistralai/mistral-small-3.1-24b-instruct:free", "google/gemma-3-1b-it:free", "google/gemma-3-12b-it:free", "google/gemma-3-27b-it:free", "meta-llama/llama-3.2-11b-vision-instruct:free"]
openrouter_model_lst = ["qwen/qwen-vl-plus", "qwen/qwen-2.5-vl-72b-instruct", "qwen/qwen-2.5-vl-7b-instruct", "mistralai/mistral-small-3.1-24b-instruct", "mistralai/pixtral-12b", "google/gemma-3-4b-it", "google/gemma-3-12b-it", "google/gemma-3-27b-it", "meta-llama/llama-3.2-11b-vision-instruct", "minimax/minimax-01",] # "meta-llama/llama-3.2-90b-vision-instruct"]

# Initialize testers
gemini_tester = GeminiCaptchaTester(
    model_lst=gemini_model_lst,
    data_path="./data",
    result_path="./results/text-captcha/test-captcha"
)

openai_tester = OpenAIAsyncCaptchaTester(
    model_lst=openai_model_lst,
    data_path="./data",
    result_path="./results/text-captcha/test-captcha"
)

openrouter_tester = OpenRouterAsyncCaptchaTester(
    model_lst=openrouter_model_lst,
    data_path="./data",
    result_path="./results/text-captcha/test-captcha"
)

async def test_openai():
    tester = OpenAIAsyncCaptchaTester(
        model_lst=[random.choice(openai_model_lst)],
        data_path="./data",
        result_path="./results/text-captcha/test/openai"
    )
    
    await tester.test_captcha(prompt_id=1, no_samples=10)
    
def test_gemini():
    tester = GeminiCaptchaTester(
        model_lst=[random.choice(gemini_model_lst)],
        data_path="./data",
        result_path="./results/text-captcha/test/gemini"
    )
    
    tester.test_captcha(prompt_id=1, no_samples=10)
    
async def test_openrouter():
    tester = OpenRouterAsyncCaptchaTester(
        model_lst=[random.choice(openrouter_model_lst)],
        data_path="./data",
        result_path="./results/text-captcha/test/openrouter"
    )
    
    await tester.test_captcha(prompt_id=1, no_samples=2)

async def main():
    # Run all testers concurrently
    for i in range(3):
        prompt_id = i+1
        gemini_tester.set_result_path(f"./results/text-captcha/test-captcha/prompt-{prompt_id}/gemini/")
        openai_tester.set_result_path(f"./results/text-captcha/test-captcha/prompt-{prompt_id}/openai/")
        openrouter_tester.set_result_path(f"./results/text-captcha/test-captcha/prompt-{prompt_id}/openrouter/")
        
        await openrouter_tester.test_captcha(prompt_id=prompt_id, no_samples=500, batch_size=50)
    
    print("All testers completed!")

if __name__ == "__main__":
    asyncio.run(main())