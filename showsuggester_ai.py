
import os
import sys
from typing import List, Tuple, Dict, Optional

import pandas as pd
from thefuzz import process

from recommender import (
    load_embeddings,
    build_user_profile_embedding,
    top_k_recommendations,
)

CSV_PATH = "tv_shows.csv"
PICKLE_PATH = "embeddings.pkl"

TOP_K = 3


def detect_title_column(df: pd.DataFrame) -> str:
    cols = {c.strip().lower(): c for c in df.columns}
    candidates = ["title", "name", "show", "show_title"]
    for c in candidates:
        if c in cols:
            return cols[c]
    raise ValueError(f"Could not detect title column. CSV columns are: {list(df.columns)}")


def load_show_titles(csv_path: str = CSV_PATH) -> List[str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find {csv_path}. Put the CSV in your project folder (same folder as this file)."
        )
    df = pd.read_csv(csv_path)
    title_col = detect_title_column(df)
    titles = (
        df[title_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )
    
    seen = set()
    uniq = []
    for t in titles:
        if t and t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


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


def fuzzy_match_shows(
    user_shows: List[str],
    all_titles: List[str],
) -> Tuple[List[str], List[Tuple[str, str, int]]]:
    matched = []
    debug_info = []

    for user_title in user_shows:
        best_match, score = process.extractOne(user_title, all_titles)
        matched.append(best_match)
        debug_info.append((user_title, best_match, int(score)))

    return matched, debug_info


def ask_yes_no(prompt: str) -> bool:
    ans = input(prompt).strip().lower()
    return ans == "y"


def format_list_for_confirmation(items: List[str]) -> str:
    return ", ".join(items)


def main():
    if not os.path.exists(PICKLE_PATH):
        print(
            f"couldn't find {PICKLE_PATH}. "
            f"run first: python build_embeddings_cache.py ."
        )
        sys.exit(1)

    title_to_embedding: Dict[str, List[float]] = load_embeddings(PICKLE_PATH)

    try:
        all_titles = load_show_titles(CSV_PATH)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    while True:
        print("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show")
        raw = input("> ").strip()

        user_shows = parse_user_input(raw)
        if len(user_shows) < 2:
            print("Please enter more than 1 show.\n")
            continue

        matched_titles, debug_info = fuzzy_match_shows(user_shows, all_titles)

        confirm_text = format_list_for_confirmation(matched_titles)
        ok = ask_yes_no(f"Making sure, do you mean {confirm_text}?(y/n) ")

        if ok:
            input_titles = matched_titles
            break

        print("Sorry about that. Lets try again, please make sure to write the names of the tv shows correctly\n")

    print("Great! Generating recommendations now…")

    try:
        user_profile = build_user_profile_embedding(input_titles, title_to_embedding)
    except ValueError as e:
        print(f"Error: {e}")
        print("Tip: If you changed tv_shows.csv after building embeddings.pkl, rebuild embeddings cache.")
        sys.exit(1)

    recs = top_k_recommendations(
        user_profile,
        title_to_embedding,
        exclude_titles=input_titles,
        k=TOP_K,
    )

    print("Here are the tv shows that i think you would love:")
    for title, score in recs:
        pct = round(score * 100)
        print(f"{title} ({pct}%)")


if __name__ == "__main__":
    main()
