import asyncio
import logging
import random

async def with_retries(model, coro_func, *args, retries=3, base_delay=2, logger=None, **kwargs):
    """
    Run an async call with exponential backoff.
    Retries for rate-limit / transient network errors.
    """
    for attempt in range(1, retries + 1):
        try:
            return await coro_func(*args, **kwargs)

        except Exception as e:
            message = str(e).lower()
            if any(x in message for x in ["rate limit", "429", "too many requests", "timeout", "overloaded"]):
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                if logger:
                    logger.warning(f"⚠️ Model [{model}] Rate-limited (attempt {attempt}/{retries}). Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                raise  # non-rate-limit error → propagate
    raise RuntimeError(f"❌ Failed after {retries} retries due to rate limits.")
