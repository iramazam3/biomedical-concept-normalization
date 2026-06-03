from tqdm import tqdm

from sapbert_encoder import (
    load_sapbert_model,
    encode_texts
)

from faiss_retriever import (
    load_faiss_resources,
    get_best_cui
)


def retrieve_cuis(
    preferred_terms
):

    tokenizer, model, device = (
        load_sapbert_model()
    )

    query_embeddings = encode_texts(
        preferred_terms,
        tokenizer,
        model,
        device
    )

    index, lookup = (
        load_faiss_resources()
    )

    predicted_cuis = []

    for embedding in tqdm(
        query_embeddings,
        desc="Retrieving CUIs"
    ):

        best_match = get_best_cui(
            embedding.reshape(1, -1),
            index,
            lookup
        )

        predicted_cuis.append(
            best_match["CUI"]
        )

    return predicted_cuis