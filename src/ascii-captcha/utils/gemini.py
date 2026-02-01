import os
from google import genai
import asyncio
from dotenv import load_dotenv

load_dotenv()

# create client once
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

async def query_gemini(model: str, prompt: str, img_b64: str | None = None):
    """Query Gemini (Google AI Studio) models via google-genai SDK."""
    def run_sync():
        if img_b64:
            # Gemini multimodal format
            image_data = {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": img_b64,
                }
            }
            response = client.models.generate_content(
                model=model,
                contents=[prompt, image_data],
            )
        else:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

        # handle safe output
        try:
            return response.text.strip()
        except AttributeError:
            return str(response)

    # run synchronous Gemini call in a thread
    return await asyncio.to_thread(run_sync)
