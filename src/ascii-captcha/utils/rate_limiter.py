import asyncio
import time

class RateLimiter:
    """
    Simple async token bucket rate limiter.
    Allows `rate` requests per `per` seconds (default: per second).
    """

    def __init__(self, rate: int, per: float = 60.0):
        self.rate = rate
        self.per = per
        self.allowance = rate
        self.last_check = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """Wait until a request can be made."""
        async with self.lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.last_check
                self.last_check = now
                self.allowance += elapsed * (self.rate / self.per)
                if self.allowance > self.rate:
                    self.allowance = self.rate

                if self.allowance >= 1.0:
                    self.allowance -= 1.0
                    return  # ✅ Allowed
                else:
                    wait_time = (1.0 - self.allowance) * (self.per / self.rate)
                    await asyncio.sleep(wait_time)
