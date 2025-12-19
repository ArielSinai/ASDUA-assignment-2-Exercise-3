
import os
import sys
import pickle
from typing import Dict, List, Tuple

import numpy as np
from thefuzz import process

PICKLE_PATH = "embeddings.pkl"
TOP_K = 5


def load_embeddings(pickle_path: str = PICKLE_PATH) -> Dict[str, List[float]]:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def parse_user_input(raw: str) -> List[str]:
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    seen = set()
    uniq = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # Distance is smaller when vectors are more similar
    return 1.0 - cosine_similarity(a, b)


def fuzzy_match_shows(user_shows: List[str], valid_titles: List[str]) -> List[str]:
    matched = []
    for s in user_shows:
        best_match, _score = process.extractOne(s, valid_titles)
        matched.append(best_match)
    return matched


def average_vector(titles: List[str], title_to_embedding: Dict[str, List[float]]) -> np.ndarray:
    vecs = []
    missing = []
    for t in titles:
        v = title_to_embedding.get(t)
        if v is None:
            missing.append(t)
        else:
            vecs.append(np.array(v, dtype=np.float32))

    if not vecs:
        raise ValueError(f"None of the input shows were found in embeddings. Missing: {missing}")

    return np.mean(vecs, axis=0).astype(np.float32)


def top_k_by_distance(
    avg_vec: np.ndarray,
    title_to_embedding: Dict[str, List[float]],
    exclude_titles: List[str],
    k: int = TOP_K,
) -> List[Tuple[str, float]]:
    exclude = set(exclude_titles)
    scored: List[Tuple[str, float]] = []

    for title, emb in title_to_embedding.items():
        if title in exclude:
            continue
        dist = cosine_distance(avg_vec, np.array(emb, dtype=np.float32))
        scored.append((title, dist))

    scored.sort(key=lambda x: x[1])  # smallest distance first
    return scored[:k]


def distances_to_percentages(top_items: List[Tuple[str, float]]) -> List[Tuple[str, int]]:
    # Map distances to 0..100 in a way that looks reasonable:
    # best (smallest distance) -> ~99, worst in top_k -> ~70 (or higher if very close).
    dists = [d for _, d in top_items]
    d_min = min(dists)
    d_max = max(dists)

    results = []
    for title, d in top_items:
        if abs(d_max - d_min) < 1e-12:
            pct = 90
        else:
            norm = (d - d_min) / (d_max - d_min)  # 0 best, 1 worst
            pct = int(round(99 - norm * 29))      # 99..70
            pct = max(1, min(99, pct))
        results.append((title, pct))

    return results


def ask_yes_no(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    return ans == "y"


def main():
    if not os.path.exists(PICKLE_PATH):
        print(f"Missing {PICKLE_PATH}. Run first: python build_embeddings_cache.py (one time).")
        sys.exit(1)

    title_to_embedding = load_embeddings(PICKLE_PATH)
    valid_titles = list(title_to_embedding.keys())

    # Step 1 + Step 2: input + fuzzy match + confirmation loop
    while True:
        print("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show")
        raw = input("> ").strip()
        user_shows = parse_user_input(raw)

        if len(user_shows) < 2:
            print("Please enter more than 1 show.\n")
            continue

        matched_titles = fuzzy_match_shows(user_shows, valid_titles)
        confirm_list = ", ".join(matched_titles)

        if ask_yes_no(f"Making sure, do you mean {confirm_list}?(y/n) "):
            input_titles = matched_titles
            break

        print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly\n")

    # Step 3
    print("Great! Generating recommendations now…")

    # Step 4: average vector + loop over all shows + cosine distance + take 5 smallest distances
    avg_vec = average_vector(input_titles, title_to_embedding)
    top_items = top_k_by_distance(avg_vec, title_to_embedding, exclude_titles=input_titles, k=TOP_K)

    # Convert distances to percentages
    top_with_pct = distances_to_percentages(top_items)

    print("Here are the tv shows that i think you would love:")
    for title, pct in top_with_pct:
        print(f"{title} ({pct}%)")


if __name__ == "__main__":
    main()
