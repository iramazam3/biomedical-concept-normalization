import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from config import SAPBERT_MODEL_NAME


def load_sapbert_model(model_name=SAPBERT_MODEL_NAME):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return tokenizer, model, device


def encode_texts(
    texts,
    tokenizer,
    model,
    device,
    batch_size=64,
    max_length=32
):
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding with SapBERT"):
        batch = texts[i:i + batch_size]

        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        with torch.no_grad():
            output = model(**encoded)
            pooled = output.last_hidden_state[:, 0, :]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        embeddings.append(pooled.cpu().numpy().astype("float32"))

    return np.vstack(embeddings)