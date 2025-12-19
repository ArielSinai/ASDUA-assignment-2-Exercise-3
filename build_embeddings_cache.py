import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

CSV_PATH = "tv_shows.csv"
PICKLE_PATH = "embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE = 64


def build_text_for_embedding(genres: str, description: str) -> str:
    g = (genres or "").strip()
    d = (description or "").replace("\n", " ").strip()
    if g and d:
        return f"Genres: {g}. Description: {d}"
    return g or d


def build_and_save_embeddings(csv_path: str = CSV_PATH, pickle_path: str = PICKLE_PATH) -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or your environment variables.")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}. Put the CSV in the project folder.")

    df = pd.read_csv(csv_path)

    required_cols = {"Title", "Description", "Genres"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must include columns: {sorted(required_cols)}. Found: {list(df.columns)}")

    df = df[["Title", "Genres", "Description"]].dropna(subset=["Title"])
    df["Title"] = df["Title"].astype(str).str.strip()
    df["Genres"] = df["Genres"].astype(str)
    df["Description"] = df["Description"].astype(str)

    titles: List[str] = df["Title"].tolist()
    texts: List[str] = [
        build_text_for_embedding(g, d) for g, d in zip(df["Genres"].tolist(), df["Description"].tolist())
    ]

    client = OpenAI(api_key=api_key)

    title_to_embedding: Dict[str, List[float]] = {}

    for i in tqdm(range(0, len(titles), BATCH_SIZE)):
        batch_titles = titles[i : i + BATCH_SIZE]
        batch_texts = texts[i : i + BATCH_SIZE]

        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
        vectors = [item.embedding for item in resp.data]

        for t, v in zip(batch_titles, vectors):
            title_to_embedding[t] = v

    with open(pickle_path, "wb") as f:
        pickle.dump(title_to_embedding, f)

    print(f"[DONE] Saved {len(title_to_embedding)} embeddings to {pickle_path} (based on Genres+Description).")


if __name__ == "__main__":
    build_and_save_embeddings()
