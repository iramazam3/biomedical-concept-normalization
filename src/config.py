import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")

SAPBERT_MODEL_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

DATA_DIR = "data"
MODEL_DIR = "models"
RESULTS_DIR = "results"

UMLS_CSV_PATH = f"{DATA_DIR}/UMLS_DATA.csv"

FAISS_INDEX_PATH = f"{MODEL_DIR}/UMLS_DATA_FAISS.index"
LOOKUP_PATH = f"{MODEL_DIR}/UMLS_DATA_LOOKUP.pkl"

TWITTER_PHRASES_PATH = f"{DATA_DIR}/twitter_symptom_phrases.txt"
CLINICAL_PHRASES_PATH = f"{DATA_DIR}/clinical_phrases.csv"

SUPPORTED_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-5",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "llama3.1:70b",
    "llama3.3:70b"
]