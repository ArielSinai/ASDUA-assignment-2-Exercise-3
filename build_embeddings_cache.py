import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

CSV_PATH = "tv_shows.csv"
PICKLE_PATH = "embeddings.pkl"
EMBEDDING_MODEL = "text-embedding-3-small"

BATCH_SIZE = 64


def _detect_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """
    מנסה לזהות עמודות של כותרת ותיאור גם אם השמות ב-CSV קצת שונים.
    """
    cols = {c.strip().lower(): c for c in df.columns}

    title_candidates = ["title", "name", "show", "show_title"]
    desc_candidates = ["description", "desc", "summary", "plot"]

    title_col = next((cols[c] for c in title_candidates if c in cols), None)
    desc_col = next((cols[c] for c in desc_candidates if c in cols), None)

    if not title_col or not desc_col:
        raise ValueError(
            f"Could not detect title/description columns. CSV columns are: {list(df.columns)}"
        )
    return title_col, desc_col


def build_and_save_embeddings(csv_path: str = CSV_PATH, pickle_path: str = PICKLE_PATH) -> None:
    if os.path.exists(pickle_path):
        print(f"[OK] {pickle_path} already exists. Not rebuilding (to avoid API calls).")
        return

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Put it in .env or your environment variables.")

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(csv_path)
    title_col, desc_col = _detect_columns(df)

    # ניקוי בסיסי
    df = df[[title_col, desc_col]].dropna()
    df[title_col] = df[title_col].astype(str).str.strip()
    df[desc_col] = df[desc_col].astype(str).str.replace("\n", " ", regex=False).str.strip()

    titles: List[str] = df[title_col].tolist()
    descriptions: List[str] = df[desc_col].tolist()

    title_to_embedding: Dict[str, List[float]] = {}

    print(f"Building embeddings for {len(titles)} shows using {EMBEDDING_MODEL} ...")

    for i in tqdm(range(0, len(titles), BATCH_SIZE)):
        batch_titles = titles[i : i + BATCH_SIZE]
        batch_texts = descriptions[i : i + BATCH_SIZE]

        
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch_texts)
        vectors = [item.embedding for item in resp.data]  

        for t, v in zip(batch_titles, vectors):
            title_to_embedding[t] = v

    with open(pickle_path, "wb") as f:
        pickle.dump(title_to_embedding, f)

    print(f"[DONE] Saved {len(title_to_embedding)} embeddings to {pickle_path}")


if __name__ == "__main__":
    build_and_save_embeddings()
