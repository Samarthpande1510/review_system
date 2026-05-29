"""
STEP 5: HYBRID RECOMMENDER
============================

CORE CONCEPT — Why Hybrid?
----------------------------
Neither model is perfect alone:

  Collaborative Filtering (MF):
    ✅ Captures complex taste patterns from millions of ratings
    ❌ Cold start: fails for new users with no ratings
    ❌ Can't recommend movies with very few ratings ("long tail")

  Content-Based:
    ✅ Works for new users (just need to know a few liked movies)
    ✅ Works for new movies (just needs text description)
    ❌ "Filter bubble": only recommends similar things, never surprising

  Hybrid = best of both worlds.

CORE CONCEPT — Weighted Blending
----------------------------------
The simplest hybrid strategy: compute a score from each model,
then combine them with a weighted average.

  hybrid_score = α * cf_score + (1 - α) * content_score

  α = 1.0  → pure collaborative filtering
  α = 0.0  → pure content-based
  α = 0.7  → 70% CF, 30% content (typical warm-start setting)

CORE CONCEPT — Cold Start vs Warm Start
-----------------------------------------
  Warm start: user has rated many movies → trust CF more (α high)
  Cold start: user has few/no ratings   → trust content more (α low)

  We set α dynamically based on how many ratings the user has:
    0 ratings    → α = 0.0
    1–5 ratings  → α = 0.3
    6–20 ratings → α = 0.6
    20+ ratings  → α = 0.9

HOW THIS FILE WORKS
--------------------
1. For a given user, get their top-rated movies
2. Get CF scores for all movies using the trained MF model
3. Get content scores using ContentRecommender
4. Blend the two score lists with weighted average
5. Return the top-N merged results
"""

import numpy as np
import pandas as pd
import torch
from typing import Optional


def get_alpha(num_ratings: int) -> float:
    """
    Dynamically set CF weight based on how many ratings the user has.
    More ratings → more trust in collaborative filtering.
    """
    if num_ratings == 0:
        return 0.0
    elif num_ratings <= 5:
        return 0.3
    elif num_ratings <= 20:
        return 0.6
    else:
        return 0.9


class HybridRecommender:
    def __init__(self,
                 mf_model,
                 content_recommender,
                 movies_df: pd.DataFrame,
                 ratings_df: pd.DataFrame,
                 device: str = "cpu"):
        """
        Args:
            mf_model             : trained MatrixFactorisation model
            content_recommender  : ContentRecommender instance
            movies_df            : movie info DataFrame
            ratings_df           : full ratings DataFrame (to look up user history)
            device               : 'cpu' or 'cuda'
        """
        self.mf_model            = mf_model
        self.content_rec         = content_recommender
        self.movies_df           = movies_df.reset_index(drop=True)
        self.ratings_df          = ratings_df
        self.device              = device
        self.mf_model.eval()     # always set to eval mode for inference

        # Precompute: all movie indices as a tensor for batch CF scoring
        self.all_movie_idxs = torch.tensor(
            movies_df["movie_idx"].values, dtype=torch.long
        ).to(device)

    def _get_cf_scores(self, user_idx: int) -> dict[int, float]:
        """
        Score ALL movies for a given user using the MF model.
        Returns dict: {movie_idx: predicted_rating}
        """
        num_movies = len(self.all_movie_idxs)

        # Repeat user_idx N times (once per movie)
        user_tensor = torch.tensor([user_idx] * num_movies,
                                    dtype=torch.long).to(self.device)

        with torch.no_grad():
            # Forward pass: score all movies at once
            scores = self.mf_model(user_tensor, self.all_movie_idxs)  # (N,)
            scores = scores.cpu().numpy()

        movie_idxs = self.all_movie_idxs.cpu().numpy()
        return dict(zip(movie_idxs.tolist(), scores.tolist()))

    def _get_user_history(self, user_idx: int) -> tuple[list[int], list[int]]:
        """
        Get the user's rated movies, split into liked (≥4) and all seen.
        Returns (liked_movie_idxs, all_seen_movie_idxs)
        """
        user_ratings = self.ratings_df[self.ratings_df["user_idx"] == user_idx]
        all_seen     = user_ratings["movie_idx"].tolist()
        liked        = user_ratings[user_ratings["rating"] >= 4]["movie_idx"].tolist()
        return liked, all_seen

    def recommend(self, user_idx: int, n: int = 10) -> pd.DataFrame:
        """
        Generate top-N hybrid recommendations for a user.

        Args:
            user_idx : the user's re-indexed ID
            n        : number of recommendations

        Returns:
            DataFrame with columns:
              [movie_idx, title, genres, cf_score, content_score, hybrid_score, alpha]
        """

        liked_movies, seen_movies = self._get_user_history(user_idx)
        num_ratings = len(seen_movies)
        alpha = get_alpha(num_ratings)

        print(f"👤 User {user_idx}: {num_ratings} ratings | α={alpha:.1f} "
              f"({'cold' if alpha < 0.5 else 'warm'} start)")

        # ── CF scores ──────────────────────────────────────────────────────────
        cf_scores = self._get_cf_scores(user_idx)   # {movie_idx: score}

        # Normalise CF scores to [0, 1] range for fair blending
        cf_vals   = np.array(list(cf_scores.values()))
        cf_min, cf_max = cf_vals.min(), cf_vals.max()
        cf_scores_norm = {
            k: (v - cf_min) / (cf_max - cf_min + 1e-8)
            for k, v in cf_scores.items()
        }

        # ── Content scores ─────────────────────────────────────────────────────
        if liked_movies:
            content_df = self.content_rec.recommend_from_liked(
                liked_movies, n=len(self.movies_df), exclude_seen=False
            )
            content_scores = dict(zip(
                content_df["movie_idx"], content_df["similarity_score"]
            ))
        else:
            # No liked movies → uniform content score (CF will dominate)
            content_scores = {idx: 0.5 for idx in cf_scores.keys()}

        # ── Blend scores ───────────────────────────────────────────────────────
        results = []
        for _, movie in self.movies_df.iterrows():
            midx = int(movie["movie_idx"])

            if midx in seen_movies:
                continue   # don't recommend already-seen movies

            cf_s      = cf_scores_norm.get(midx, 0.0)
            content_s = content_scores.get(midx, 0.0)
            hybrid_s  = alpha * cf_s + (1 - alpha) * content_s

            results.append({
                "movie_idx":     midx,
                "title":         movie["title"],
                "genres":        movie["genres"],
                "cf_score":      round(cf_s, 4),
                "content_score": round(content_s, 4),
                "hybrid_score":  round(hybrid_s, 4),
                "alpha":         alpha,
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("hybrid_score", ascending=False)
        return results_df.head(n).reset_index(drop=True)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from models.mf_model import MatrixFactorisation
    from models.content_model import build_content_embeddings, ContentRecommender
    from data.load_data import load_movielens
    DATA_DIR   = "."
    MODEL_PATH = "models/mf_model.pt"

    train_df, test_df, movies_df, num_users, num_movies = load_movielens(DATA_DIR)
    all_ratings = pd.concat([train_df, test_df])

    # Load trained MF model
    device = "cpu"
    mf_model = MatrixFactorisation(num_users, num_movies, embedding_dim=32)
    mf_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Build content model
    embeddings, movies_df = build_content_embeddings(movies_df)
    content_rec = ContentRecommender(embeddings, movies_df)

    # Build hybrid recommender
    hybrid = HybridRecommender(mf_model, content_rec, movies_df, all_ratings, device)

    # Test on a warm-start user (user 0 has many ratings)
    recs = hybrid.recommend(user_idx=0, n=10)
    print("\n🎬 Top 10 Hybrid Recommendations for User 0:")
    print(recs[["title", "cf_score", "content_score", "hybrid_score"]].to_string(index=False))