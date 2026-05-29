"""
STEP 3: DATASET + TRAINING LOOP
=================================

CORE CONCEPT — torch.utils.data.Dataset
-----------------------------------------
PyTorch wants data served through a Dataset object. You subclass it and
implement two methods:
  __len__()       → how many samples total?
  __getitem__(i)  → return the i-th sample as tensors

DataLoader wraps the Dataset and handles:
  - Batching (grouping samples into mini-batches)
  - Shuffling (randomise order each epoch)
  - Parallel loading (num_workers)

CORE CONCEPT — Training Loop
------------------------------
For each epoch:
  1. Grab a mini-batch of (user, movie, rating)
  2. Forward pass  → model predicts ratings
  3. Compute loss  → MSE between prediction and actual rating
  4. Backward pass → compute gradients (autograd does this)
  5. Optimizer step → update weights using gradients
  6. Zero gradients → reset for next batch (IMPORTANT — PyTorch accumulates!)

CORE CONCEPT — MSE Loss
-------------------------
  MSE = mean( (predicted - actual)² )
  We minimise this so predictions get closer to real ratings.
  RMSE = sqrt(MSE) is more interpretable (same units as rating scale 1–5).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np


# ── Dataset ──────────────────────────────────────────────────────────────────

class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        """
        df must have columns: user_idx, movie_idx, rating
        We convert to numpy first for fast indexing.
        """
        self.users   = torch.tensor(df["user_idx"].values,  dtype=torch.long)
        self.movies  = torch.tensor(df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values,    dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        # Returns one sample — DataLoader will collate many of these into a batch
        return self.users[idx], self.movies[idx], self.ratings[idx]


# ── Training & Evaluation ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()       # activates dropout/batchnorm if present (good habit)
    total_loss = 0.0

    for users, movies, ratings in loader:
        # Move data to the same device as the model (CPU or GPU)
        users   = users.to(device)
        movies  = movies.to(device)
        ratings = ratings.to(device)

        optimizer.zero_grad()           # 1. clear old gradients
        preds = model(users, movies)    # 2. forward pass
        loss  = criterion(preds, ratings)  # 3. compute loss
        loss.backward()                 # 4. backprop
        optimizer.step()                # 5. update weights

        total_loss += loss.item() * len(ratings)  # accumulate weighted loss

    return np.sqrt(total_loss / len(loader.dataset))  # return RMSE


def evaluate(model, loader, criterion, device):
    model.eval()        # disables dropout/batchnorm
    total_loss = 0.0

    with torch.no_grad():   # no gradients needed for evaluation → faster + less memory
        for users, movies, ratings in loader:
            users   = users.to(device)
            movies  = movies.to(device)
            ratings = ratings.to(device)

            preds = model(users, movies)
            loss  = criterion(preds, ratings)
            total_loss += loss.item() * len(ratings)

    return np.sqrt(total_loss / len(loader.dataset))


# ── Main Training Script ──────────────────────────────────────────────────────

def train(train_df, test_df, num_users, num_movies,
          embedding_dim=32, epochs=5, batch_size=1024, lr=1e-3,
          save_path="models/mf_model.pt"):

    # Import here to avoid circular imports
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from mf_model import MatrixFactorisation

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Training on: {device}")

    # Datasets & loaders
    train_ds = RatingsDataset(train_df)
    test_ds  = RatingsDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)

    # Model, loss, optimiser
    model     = MatrixFactorisation(num_users, num_movies, embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Learning rate scheduler — halves LR if val loss doesn't improve for 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    print(f"\n{'Epoch':>6} {'Train RMSE':>12} {'Val RMSE':>10}")
    print("-" * 32)

    best_val_rmse = float("inf")

    for epoch in range(1, epochs + 1):
        train_rmse = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_rmse   = evaluate(model, test_loader, criterion, device)
        scheduler.step(val_rmse)

        print(f"{epoch:>6} {train_rmse:>12.4f} {val_rmse:>10.4f}")

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ Best Val RMSE: {best_val_rmse:.4f}  →  saved to {save_path}")
    return model


# ── Run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.load_data import load_movielens

    DATA_DIR = "."   # ← change to folder containing .dat files

    train_df, test_df, movies_df, num_users, num_movies = load_movielens(DATA_DIR)

    train(
        train_df, test_df,
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=32,
        epochs=5,
        batch_size=1024,
        lr=1e-3,
        save_path="models/mf_model.pt",
    )