import os
import pandas as pd
from tqdm import tqdm

from llm_normalizer import normalize_phrase
from cui_retrieval import retrieve_cuis
from config import RESULTS_DIR


MODEL_NAME = "gpt-4o"

INPUT_FILE = "data/symptoms.txt"
OUTPUT_FILE = f"{RESULTS_DIR}/predictions.csv"


def load_phrases(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        phrases = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return phrases


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    phrases = load_phrases(INPUT_FILE)

    print(f"Loaded {len(phrases)} phrases")
    print(f"Using model: {MODEL_NAME}")

    preferred_terms = []

    for phrase in tqdm(phrases, desc="Generating preferred terms"):
        preferred_term = normalize_phrase(
            phrase=phrase,
            model_name=MODEL_NAME
        )

        preferred_terms.append(preferred_term)

    predicted_cuis = retrieve_cuis(preferred_terms)

    results = pd.DataFrame({
        "Phrase": phrases,
        "Preferred Term": preferred_terms,
        "Predicted CUI": predicted_cuis
    })

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"Saved predictions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()