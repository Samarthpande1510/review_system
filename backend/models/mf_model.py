"""
STEP 2: MATRIX FACTORISATION MODEL
====================================

CORE CONCEPT — Embeddings
--------------------------
An nn.Embedding(N, D) is just a lookup table:
  - N rows  = one row per user (or movie)
  - D cols  = the learnable "taste" vector for that user

  user_embedding[42]  → a vector like [0.3, -1.2, 0.8, ...]  (D numbers)
  movie_embedding[99] → a vector like [0.1,  0.9, 0.4, ...]  (D numbers)

CORE CONCEPT — Dot Product as Predicted Rating
-----------------------------------------------
  predicted_rating = sum(user_vec * movie_vec)

  If both vectors align (same signs, large values) → high predicted rating
  If they don't align → low predicted rating

  Training adjusts both vectors so this dot product ≈ actual rating.

CORE CONCEPT — Forward Pass
-----------------------------
  input: user index (int), movie index (int)
  output: scalar predicted rating

CORE CONCEPT — Bias Terms
--------------------------
  Some users always rate high. Some movies are universally loved.
  We add a scalar bias per user and per movie to capture this:
  prediction = dot(user, movie) + user_bias + movie_bias + global_mean
"""

import torch
import torch.nn as nn


class MatrixFactorisation(nn.Module):
    def __init__(self, num_users: int, num_movies: int, embedding_dim: int = 32):
        super().__init__()

        # Embedding tables — randomly initialised, learned during training
        self.user_embeddings  = nn.Embedding(num_users,  embedding_dim)
        self.movie_embeddings = nn.Embedding(num_movies, embedding_dim)

        # Bias: one scalar per user / movie
        self.user_bias  = nn.Embedding(num_users,  1)
        self.movie_bias = nn.Embedding(num_movies, 1)

        # Initialise weights small so training is stable
        nn.init.normal_(self.user_embeddings.weight,  std=0.01)
        nn.init.normal_(self.movie_embeddings.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)

    def forward(self, user_idx: torch.Tensor, movie_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            user_idx  : (batch_size,) int tensor of user indices
            movie_idx : (batch_size,) int tensor of movie indices
        Returns:
            preds     : (batch_size,) float tensor of predicted ratings
        """

        # Look up embedding vectors for the batch
        u = self.user_embeddings(user_idx)   # (batch, dim)
        m = self.movie_embeddings(movie_idx) # (batch, dim)

        # Element-wise multiply then sum along dim=1  →  dot product per sample
        dot = torch.sum(u * m, dim=1)        # (batch,)

        # Add biases (squeeze removes the trailing dim=1)
        ub = self.user_bias(user_idx).squeeze()    # (batch,)
        mb = self.movie_bias(movie_idx).squeeze()  # (batch,)

        return dot + ub + mb