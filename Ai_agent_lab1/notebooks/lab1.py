import os
import time
from dotenv import load_dotenv
from groq import Groq
from google import genai
from openai import OpenAI

load_dotenv()

PROMPT = "Объясни разницу между машинным обучением и LLM"


def ask_groq():
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    start = time.perf_counter()

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": PROMPT}]
    )

    elapsed = time.perf_counter() - start

    return {
        "model": response.model,
        "time": elapsed,
        "tokens": response.usage.total_tokens,
        "answer": response.choices[0].message.content
    }



def ask_gemini():
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    start = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=PROMPT
    )

    elapsed = time.perf_counter() - start

    return {
        "model": "gemini-3.8-flash",
        "time": elapsed,
        "tokens": getattr(response.usage_metadata, "total_token_count", 0),
        "answer": response.text
    }


def ask_openrouter():
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    start = time.perf_counter()

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": PROMPT}]
    )

    elapsed = time.perf_counter() - start

    return {
        "model": response.model,
        "time": elapsed,
        "tokens": response.usage.total_tokens,
        "answer": response.choices[0].message.content
    }


if __name__ == "__main__":
    results = [
        ask_groq(),
        ask_gemini(),
        ask_openrouter()
    ]

    for result in results:
        print("\n" + "=" * 60)
        print("Model:", result["model"])
        print("Time:", round(result["time"], 2), "sec")
        print("Tokens:", result["tokens"])
        print("Answer:")
        print(result["answer"])
