import anthropic
import asyncio

client = anthropic.Anthropic()

async def query_anthropic(model: str, prompt: str, img_b64: str | None = None):
    """Query Anthropic (Claude) models with multimodal input."""
    def run_sync():
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
                + (
                    [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        }
                    ]
                    if img_b64
                    else []
                ),
            }
        ]
        response = client.messages.create(model=model, max_tokens=1024, messages=messages)
        return response.content[0].text if response.content else ""

    return await asyncio.to_thread(run_sync)
