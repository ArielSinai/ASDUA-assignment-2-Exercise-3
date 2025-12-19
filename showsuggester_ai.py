import os
from recommender import load_embeddings, build_user_profile_embedding, top_k_recommendations


def main():
    if not os.path.exists("embeddings.pkl"):
        print("couldn't find embeddings.pkl. run first: python build_embeddings_cache.py")
        return

    title_to_embedding = load_embeddings("embeddings.pkl")

    print("Which TV shows did you really like watching? Separate them by a comma. Make sure to enter more than 1 show")
    raw = input("> ").strip()
    shows = [s.strip() for s in raw.split(",") if s.strip()]

    if len(shows) < 2:
        print("Please enter more than 1 show.")
        return

    profile = build_user_profile_embedding(shows, title_to_embedding)
    recs = top_k_recommendations(profile, title_to_embedding, exclude_titles=shows, k=3)

    print("\nHere are the tv shows that i think you would love:")
    for title, score in recs:
        pct = round(score * 100)
        print(f"{title} ({pct}%)")


if __name__ == "__main__":
    main()
