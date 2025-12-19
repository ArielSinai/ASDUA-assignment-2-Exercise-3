import os
import sys
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from annoy import AnnoyIndex
from thefuzz import process, fuzz

PICKLE_PATH = "embeddings.pkl"

ANNOY_INDEX_PATH = "annoy_index.ann"
ANNOY_META_PATH = "annoy_meta.json"

TOP_K = 5
CANDIDATES = 50
N_TREES = 50
MIN_SCORE = 75


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


def ask_yes_no(prompt: str) -> bool:
    return input(prompt).strip().lower() == "y"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine_similarity(a, b)


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


def build_acronym_map(titles: List[str]) -> Dict[str, str]:
    acronym_map: Dict[str, str] = {}
    for t in titles:
        words = [w for w in t.split() if w and w[0].isalnum()]
        ac = "".join(w[0] for w in words).upper()
        if len(ac) >= 3 and ac not in acronym_map:
            acronym_map[ac] = t
    return acronym_map


def fuzzy_match_shows(
    user_shows: List[str],
    valid_titles: List[str],
    acronym_map: Dict[str, str],
) -> Tuple[List[str], List[Tuple[str, str, int]]]:
    matched = []
    debug = []
    for s in user_shows:
        key = s.replace(".", "").replace("-", "").strip().upper()
        if key in acronym_map:
            best, score = acronym_map[key], 100
        else:
            best, score = process.extractOne(s, valid_titles, scorer=fuzz.WRatio)
        matched.append(best)
        debug.append((s, best, int(score)))
    return matched, debug


def distances_to_percentages(top_items: List[Tuple[str, float]]) -> List[Tuple[str, int]]:
    dists = [d for _, d in top_items]
    d_min = min(dists)
    d_max = max(dists)

    out = []
    for title, d in top_items:
        if abs(d_max - d_min) < 1e-12:
            pct = 90
        else:
            norm = (d - d_min) / (d_max - d_min)  # 0 best, 1 worst
            pct = int(round(99 - norm * 29))      # 99..70
            pct = max(1, min(99, pct))
        out.append((title, pct))
    return out


def build_or_load_annoy_index(
    title_to_embedding: Dict[str, List[float]],
    index_path: str = ANNOY_INDEX_PATH,
    meta_path: str = ANNOY_META_PATH,
) -> Tuple[AnnoyIndex, Dict[int, str], Dict[str, int]]:
    titles = list(title_to_embedding.keys())
    dim = len(title_to_embedding[titles[0]])

    index_file = Path(index_path)
    meta_file = Path(meta_path)

    if index_file.exists() and meta_file.exists():
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("dim") != dim:
            index_file.unlink(missing_ok=True)
            meta_file.unlink(missing_ok=True)
        else:
            id_to_title = {int(k): v for k, v in meta["id_to_title"].items()}
            title_to_id = {v: int(k) for k, v in meta["id_to_title"].items()}
            annoy = AnnoyIndex(dim, "angular")
            annoy.load(str(index_file))
            return annoy, id_to_title, title_to_id

    id_to_title: Dict[int, str] = {}
    title_to_id: Dict[str, int] = {}

    annoy = AnnoyIndex(dim, "angular")

    for idx, title in enumerate(titles):
        vec = np.array(title_to_embedding[title], dtype=np.float32)
        annoy.add_item(idx, vec.tolist())
        id_to_title[idx] = title
        title_to_id[title] = idx

    annoy.build(N_TREES)
    annoy.save(str(index_file))

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dim": dim,
                "id_to_title": {str(k): v for k, v in id_to_title.items()},
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return annoy, id_to_title, title_to_id


def recommend_with_annoy(
    avg_vec: np.ndarray,
    annoy: AnnoyIndex,
    id_to_title: Dict[int, str],
    title_to_embedding: Dict[str, List[float]],
    exclude_titles: List[str],
    k: int = TOP_K,
    candidates: int = CANDIDATES,
) -> List[Tuple[str, float]]:
    exclude = set(exclude_titles)

    ids = annoy.get_nns_by_vector(avg_vec.tolist(), candidates, include_distances=False)

    scored: List[Tuple[str, float]] = []
    for idx in ids:
        title = id_to_title.get(idx)
        if not title or title in exclude:
            continue
        vec = np.array(title_to_embedding[title], dtype=np.float32)
        dist = cosine_distance(avg_vec, vec)
        scored.append((title, dist))

    if len(scored) < k:
        for title, emb in title_to_embedding.items():
            if title in exclude:
                continue
            vec = np.array(emb, dtype=np.float32)
            dist = cosine_distance(avg_vec, vec)
            scored.append((title, dist))

    scored.sort(key=lambda x: x[1])
    return scored[:k]


def main():
    if not os.path.exists(PICKLE_PATH):
        print(f"Missing {PICKLE_PATH}. Run first: python build_embeddings_cache.py (one time).")
        sys.exit(1)

    title_to_embedding = load_embeddings(PICKLE_PATH)
    valid_titles = list(title_to_embedding.keys())
    acronym_map = build_acronym_map(valid_titles)

    annoy, id_to_title, _title_to_id = build_or_load_annoy_index(title_to_embedding)

    while True:
        print("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show")
        raw = input("> ").strip()
        user_shows = parse_user_input(raw)

        if len(user_shows) < 2:
            print("Please enter more than 1 show.\n")
            continue

        matched_titles, debug_info = fuzzy_match_shows(user_shows, valid_titles, acronym_map)

        if any(score < MIN_SCORE for _, _, score in debug_info):
            print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly\n")
            continue

        confirm_list = ", ".join(matched_titles)
        if ask_yes_no(f"Making sure, do you mean {confirm_list}?(y/n) "):
            input_titles = matched_titles
            break

        print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly\n")

    print("Great! Generating recommendations now…")

    avg_vec = average_vector(input_titles, title_to_embedding)
    top_items = recommend_with_annoy(
        avg_vec,
        annoy,
        id_to_title,
        title_to_embedding,
        exclude_titles=input_titles,
        k=TOP_K,
        candidates=CANDIDATES,
    )

    top_with_pct = distances_to_percentages(top_items)

    print("Here are the tv shows that i think you would love:")
    for title, pct in top_with_pct:
        print(f"{title} ({pct}%)")


if __name__ == "__main__":
    main()
