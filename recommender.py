import pickle
from typing import Dict, List, Tuple

import numpy as np


def load_embeddings(pickle_path: str = "embeddings.pkl") -> Dict[str, List[float]]:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def build_user_profile_embedding(
    user_show_titles: List[str],
    title_to_embedding: Dict[str, List[float]],
) -> np.ndarray:
    vectors = []
    missing = []
    for t in user_show_titles:
        if t in title_to_embedding:
            vectors.append(np.array(title_to_embedding[t], dtype=np.float32))
        else:
            missing.append(t)

    if not vectors:
        raise ValueError(f"None of the user shows were found in embeddings. Missing: {missing}")

    profile = np.mean(vectors, axis=0)
    return profile.astype(np.float32)


def top_k_recommendations(
    user_profile: np.ndarray,
    title_to_embedding: Dict[str, List[float]],
    exclude_titles: List[str],
    k: int = 5,
) -> List[Tuple[str, float]]:
    exclude_set = set(exclude_titles)
    scored = []

    for title, emb in title_to_embedding.items():
        if title in exclude_set:
            continue
        sim = cosine_similarity(user_profile, np.array(emb, dtype=np.float32))
        scored.append((title, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
