import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1",api_key=os.environ["OPENROUTER_API_KEY"])

async def query_openrouter(provider: str, model: str, prompt: str, img_b64: str | None = None) -> str:
    """
    Query OpenRouter models via the official Responses API.

    Args:
        model: e.g., "gpt-4o-mini" or "gpt-4o"
        prompt: The text prompt to send.
        img_b64: Optional base64-encoded image (as string) for multimodal input.

    Returns:
        str: The model's textual response.
    """
    if img_b64:
        content = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img_b64}",
                        "detail": "high",
                    },
                ],
            }
        ]
    else:
        content = [
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ]

    try:
        response = await client.responses.create(
            model=f"{provider}/{model}",
            input=content,
            max_output_tokens=512,
        )
        return response.output_text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"
