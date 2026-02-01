import asyncio
from util import ASCIICaptchaTester

STRICT_PROMPT = (
    "You are taking part in a research experiment on text pattern recognition.\n"
    "The input shown above is a block of ASCII characters that visually represents some alphanumeric text.\n"
    "Extract and return only the exact sequence of visible alphanumeric characters you can read from it.\n"
    "Do not mention images, descriptions, explanations, or uncertainty.\n"
    "Output only the recognized text as a single continuous string, with no spaces, punctuation, or commentary.\n"
    "If the characters are ambiguous, make your best data-driven guess and output only that guess."
)

render_as_image = True
PROMPT = STRICT_PROMPT

async def main():
    tester = ASCIICaptchaTester(
        model_list=[
            # "google/gemini-2.5-flash",
            # "google/gemini-2.5-pro",
            # "google/gemini-2.5-flash-lite",
            
            # "openai/gpt-5-chat-latest",
            # "openai/gpt-4.1-mini",
            # "openai/gpt-4.1-nano",

            # "anthropic/claude-sonnet-4.5",
            # "anthropic/claude-sonnet-4",
            # "anthropic/claude-haiku-4.5",

            # "meta-llama/llama-4-maverick",
            # "meta-llama/llama-4-scout",

            # "qwen/qwen3-vl-235b-a22b-instruct",
            # "qwen/qwen3-vl-30b-a3b-instruct",
            # "qwen/qwen3-vl-8b-instruct",

            # # text-only
            # "deepseek/deepseek-chat-v3-0324",
            # "deepseek/deepseek-chat-v3.1",
            # "deepseek/deepseek-v3.2-exp",
            # "deepseek/deepseek-r1-0528",

            "qwen/qwen3-vl-8b-instruct",
            "google/gemini-2.5-pro",
            "anthropic/claude-haiku-4.5",
            "qwen/qwen3-vl-30b-a3b-instruct",
            "anthropic/claude-sonnet-4.5",
            "anthropic/claude-sonnet-4",
            "meta-llama/llama-4-scout",
            # "deepseek/deepseek-chat-v3.1",
            "qwen/qwen3-vl-235b-a22b-instruct",
            "google/gemini-3-flash-preview",
        ],
        result_path=f"results/ascii-final-1/{'image' if render_as_image else 'text'}/",
        data_path="data/ascii-captcha",
        concurrency_limit=50,
    )

    # await tester.debug_test_mode("Hi! Say hello! Also, briefly say what is shown in the image.")

    await tester.run_all_models(
        n=250,
        prompt=PROMPT,
        render_as_image=render_as_image,
    )

asyncio.run(main())
