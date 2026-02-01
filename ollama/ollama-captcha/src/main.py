from ollama import AsyncClient

from util import ASCIICaptchaTester

import asyncio

host = "http://127.0.0.1:11434/" # change if needed
client = AsyncClient(host)

model_lst = ["llava:latest", "llava-phi3:latest", "gemma3:latest", "qwen2.5vl:3b"] # change this to larger more powerful models
# model_lst = ["llava:34b", "gemma3:27b", "qwen2.5vl:32b", "qwen2.5vl:72b"]

tester = ASCIICaptchaTester(client)

message_lst_normal = [{
    "role": "system",
    "content": "You must analyse the provided text or image and extract the exact case-sensitive alphanumeric text. \
    Return only the solution as plain text with no additional words, symbols, formatting, or explanation. \
    The text contains only alphanumeric characters with no spaces. \
    Your response must strictly be the predicted text only."
}]

message_lst_sequence = [{
    "role": "system",
    "content": (
        "Extract the exact case-sensitive alphanumeric string from the text or image. "
        "Output ONLY the string: no spaces, punctuation, or formatting.\n\n"
        "Steps (internal, do not output): "
        "1) Trim empty rows/cols. "
        "2) Split glyphs by blank columns or narrow joins. "
        "3) For each segment, classify by strokes (bars, loops, diagonals), holes, and open vs closed shapes. "
        "Resolve lookalikes: O/0, I/1/l, S/5, Z/2, B/8, G/6, Q=O+tail, etc. "
        "4) Use ascenders/descenders for case. "
        "5) Concatenate left→right, ensuring only [A-Za-z0-9].\n\n"
        "Return ONLY the final string."
    )
}]

async def run_all_tests():
    await tester.run_test_suite(name="normal", message_lst=message_lst_normal, model_lst=model_lst)
    # await tester.run_test_suite(name="sequence", message_lst=message_lst_sequence, model_lst=model_lst)

    return

if __name__ == "__main__":
    asyncio.run(run_all_tests())