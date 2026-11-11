import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter

load_dotenv()

_rate_limiter = InMemoryRateLimiter(requests_per_second=0.4)

def get_llm(temperature: float = 0.3):
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise ValueError("MISTRAL_API_KEY is missing.")

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=temperature,
        rate_limiter=_rate_limiter,
        max_retries=6,
    )