import pickle
import pandas as pd
import faiss

from sapbert_encoder import load_sapbert_model, encode_texts
from config import UMLS_CSV_PATH, FAISS_INDEX_PATH, LOOKUP_PATH


def build_umls_index():
    print("Loading UMLS data...")

    df = pd.read_csv(UMLS_CSV_PATH)

    required_columns = ["CUI", "STR", "TTY", "STY"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    umls_terms = df["STR"].astype(str).tolist()

    print(f"Loaded {len(umls_terms):,} UMLS terms")

    tokenizer, model, device = load_sapbert_model()

    embeddings = encode_texts(
        umls_terms,
        tokenizer,
        model,
        device,
        batch_size=128,
        max_length=32
    )

    print("Embedding shape:", embeddings.shape)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"FAISS index built with {index.ntotal:,} vectors")

    faiss.write_index(index, FAISS_INDEX_PATH)

    lookup = {
        "cuis": df["CUI"].astype(str).tolist(),
        "terms": df["STR"].astype(str).tolist(),
        "ttys": df["TTY"].astype(str).tolist(),
        "stys": df["STY"].astype(str).tolist()
    }

    with open(LOOKUP_PATH, "wb") as f:
        pickle.dump(lookup, f)

    print("Saved FAISS index and lookup table")


if __name__ == "__main__":
    build_umls_index()