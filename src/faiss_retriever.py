import pickle
import faiss

from config import FAISS_INDEX_PATH, LOOKUP_PATH


def load_faiss_resources():
    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(LOOKUP_PATH, "rb") as f:
        lookup = pickle.load(f)

    return index, lookup


def search_umls(
    query_embedding,
    index,
    lookup,
    k=5
):
    distances, indices = index.search(query_embedding, k)

    results = []

    for score, idx in zip(distances[0], indices[0]):
        results.append({
            "CUI": lookup["cuis"][idx],
            "Term": lookup["terms"][idx],
            "TTY": lookup["ttys"][idx],
            "STY": lookup["stys"][idx],
            "Score": float(score)
        })

    return results


def get_best_cui(
    query_embedding,
    index,
    lookup,
    k=5,
    filter_sign_symptom=True
):
    candidates = search_umls(
        query_embedding,
        index,
        lookup,
        k=k
    )

    if filter_sign_symptom:
        symptom_candidates = [
            c for c in candidates
            if str(c["STY"]).strip().lower() == "sign or symptom"
        ]

        if symptom_candidates:
            return max(symptom_candidates, key=lambda x: x["Score"])

    return max(candidates, key=lambda x: x["Score"])