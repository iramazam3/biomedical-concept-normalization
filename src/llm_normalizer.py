import json
import requests

from openai import OpenAI
from google import genai

from config import OPENAI_API_KEY, GOOGLE_API_KEY, OLLAMA_URL


PROMPT_TEMPLATE = """
Normalize the following medical symptom phrase into standard medical terminology.

Provide the single most appropriate, top-level preferred term.

Respond only with the preferred term.

Phrase: "{phrase}"

Preferred term:
"""


def build_prompt(phrase):
    return PROMPT_TEMPLATE.format(phrase=phrase)


def normalize_openai(
    phrase,
    model_name="gpt-4o"
):
    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env file.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    prompt = build_prompt(phrase)

    response = client.responses.create(
        model=model_name,
        input=prompt,
        temperature=0
    )

    return response.output_text.strip()


def normalize_gemini(
    phrase,
    model_name="gemini-2.0-flash"
):
    if not GOOGLE_API_KEY:
        raise ValueError("Missing GOOGLE_API_KEY in .env file.")

    client = genai.Client(api_key=GOOGLE_API_KEY)

    prompt = build_prompt(phrase)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text.strip()


def normalize_ollama(
    phrase,
    model_name="llama3.1:70b"
):
    prompt = build_prompt(phrase)

    payload = {
        "model": model_name,
        "prompt": prompt
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        stream=True,
        timeout=120
    )

    output = ""

    for line in response.iter_lines():
        if line:
            chunk = json.loads(line.decode("utf-8"))
            output += chunk.get("response", "")

    return output.strip()


def normalize_phrase(
    phrase,
    model_name
):
    if model_name.startswith("gpt"):
        return normalize_openai(phrase, model_name)

    if model_name.startswith("gemini"):
        return normalize_gemini(phrase, model_name)

    if model_name.startswith("llama"):
        return normalize_ollama(phrase, model_name)

    raise ValueError(f"Unsupported model: {model_name}")