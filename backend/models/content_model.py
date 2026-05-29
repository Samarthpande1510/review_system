"""
STEP 4: CONTENT-BASED MODEL (Sentence-Transformers)
=====================================================

CORE CONCEPT — Sentence Embeddings
------------------------------------
Sentence-Transformers is a pretrained model that converts any text string
into a dense vector (embedding) of fixed size (e.g. 384 numbers).

  "Toy Story (1995) | Animation|Children's|Comedy"
       ↓  model encodes this
  [0.21, -0.13, 0.87, ..., 0.04]   ← 384-dimensional vector

Movies with similar descriptions will have similar vectors.
"Toy Story" and "A Bug's Life" → vectors close together in space.
"Toy Story" and "The Godfather" → vectors far apart.

CORE CONCEPT — Cosine Similarity vs Dot Product
-------------------------------------------------
Two ways to measure how similar two vectors u and v are:

  Dot product:      sum(u * v)
    → affected by vector magnitude (length). Longer vectors score higher
      even if direction is the same.

  Cosine similarity: dot(u, v) / (||u|| * ||v||)
    → normalises by length. Pure measure of direction/angle.
    → value in [-1, 1], where 1 = identical direction, 0 = orthogonal

For content similarity, cosine is preferred because we care about
*meaning* alignment, not vector scale.

CORE CONCEPT — Cold Start Problem
-----------------------------------
Collaborative filtering (MF) needs a user's past ratings to make
predictions. New users have NO ratings → it can't work. This is
called the "cold start" problem.

Content-based doesn't need ratings — it just looks at movie text.
So for cold-start users, we fall back entirely to content-based.

HOW THIS FILE WORKS
--------------------
1. Build a text description for each movie: "Title | Genre1|Genre2"
2. Encode all descriptions into embeddings with SentenceTransformer
3. Save the embeddings matrix (movies × 384) to disk
4. At query time: encode the user's liked movies, average their
   embeddings, then find movies closest to that average.
"""

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle


# ── Build & save embeddings ───────────────────────────────────────────────────

def build_content_embeddings(movies_df: pd.DataFrame,
                              save_path: str = "models/content_embeddings.pkl",
                              model_name: str = "all-MiniLM-L6-v2"):
    """
    Encode every movie's title + genres into a sentence embedding.

    'all-MiniLM-L6-v2' is a small, fast model (80MB) that produces
    384-dimensional embeddings. Good balance of speed and quality.

    Args:
        movies_df : DataFrame with columns [movie_idx, title, genres]
        save_path : where to cache the embeddings (encoding is slow)
        model_name: which Sentence-Transformer model to use

    Returns:
        embeddings : np.ndarray of shape (num_movies, 384)
        movies_df  : same df, now with an 'embedding_idx' column
    """

    if os.path.exists(save_path):
        print(f"📦 Loading cached embeddings from {save_path}")
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        return data["embeddings"], data["movies_df"]

    print(f"🔄 Building content embeddings with {model_name}...")
    encoder = SentenceTransformer(model_name)

    # Build a text description for each movie
    # Format: "Toy Story (1995) | Animation Children's Comedy"
    movies_df = movies_df.copy()
    movies_df["text"] = (
        movies_df["title"] + " | " +
        movies_df["genres"].str.replace("|", " ", regex=False)
    )

    texts = movies_df["text"].tolist()

    # encode() returns np.ndarray of shape (N, 384)
    # show_progress_bar=True prints a tqdm bar — useful since this takes ~30s
    embeddings = encoder.encode(texts, show_progress_bar=True, batch_size=64)

    # Normalise embeddings to unit length so cosine sim = dot product later
    # This is a common optimisation — avoids the division in cosine formula
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms   # shape: (num_movies, 384)

    # Save to disk so we don't re-encode every run
    with open(save_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "movies_df": movies_df}, f)

    print(f"✅ Saved embeddings: {embeddings.shape} → {save_path}")
    return embeddings, movies_df


# ── Content-based recommender ─────────────────────────────────────────────────

class ContentRecommender:
    def __init__(self, embeddings: np.ndarray, movies_df: pd.DataFrame):
        """
        Args:
            embeddings : (num_movies, 384) normalised embedding matrix
            movies_df  : DataFrame indexed by position matching embeddings rows
        """
        self.embeddings = embeddings          # (num_movies, 384)
        self.movies_df  = movies_df.reset_index(drop=True)

        # Build a fast lookup: movie_idx → row position in embeddings matrix
        self.idx_to_row = {
            int(row["movie_idx"]): i
            for i, row in self.movies_df.iterrows()
        }

    def recommend_from_liked(self,
                              liked_movie_idxs: list[int],
                              n: int = 10,
                              exclude_seen: bool = True) -> pd.DataFrame:
        """
        Given a list of movie_idxs the user liked, recommend similar movies.

        Strategy:
          1. Get embeddings for liked movies
          2. Average them → a "taste vector" for this user
          3. Find movies with highest cosine similarity to the taste vector

        Args:
            liked_movie_idxs : list of movie_idx values the user rated highly
            n                : number of recommendations to return
            exclude_seen     : don't recommend movies the user already rated

        Returns:
            DataFrame with columns [movie_idx, title, genres, similarity_score]
        """

        # Convert movie_idxs to row positions in embeddings matrix
        rows = [self.idx_to_row[idx] for idx in liked_movie_idxs
                if idx in self.idx_to_row]

        if not rows:
            # Fallback: return most popular movies (just first n)
            return self.movies_df.head(n)[["movie_idx", "title", "genres"]]

        # Average the liked movies' embeddings → user "taste" vector
        # Shape: (1, 384) — the [np.newaxis] adds a batch dimension for cosine_similarity
        taste_vector = self.embeddings[rows].mean(axis=0)[np.newaxis, :]

        # Cosine similarity of taste_vector vs ALL movies
        # Since embeddings are unit-normalised, this is just a dot product
        # scores shape: (1, num_movies) → squeeze to (num_movies,)
        scores = cosine_similarity(taste_vector, self.embeddings)[0]

        # Sort by score descending
        ranked_rows = np.argsort(scores)[::-1]

        results = []
        for row in ranked_rows:
            movie = self.movies_df.iloc[row]
            if exclude_seen and int(movie["movie_idx"]) in liked_movie_idxs:
                continue
            results.append({
                "movie_idx":        int(movie["movie_idx"]),
                "title":            movie["title"],
                "genres":           movie["genres"],
                "similarity_score": float(scores[row]),
            })
            if len(results) >= n:
                break

        return pd.DataFrame(results)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.load_data import load_movielens

    _, _, movies_df, _, _ = load_movielens(".")

    embeddings, movies_df = build_content_embeddings(movies_df)
    recommender = ContentRecommender(embeddings, movies_df)

    # Simulate a user who liked Toy Story (idx 0) and Jumanji (idx 1)
    liked = [0, 1]
    recs = recommender.recommend_from_liked(liked, n=5)
    print("\nContent-based recommendations:")
    print(recs[["title", "similarity_score"]].to_string(index=False))