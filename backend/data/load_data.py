import pandas as pd
from sklearn.model_selection import train_test_split


def load_movielens(data_dir: str):
    ratings = pd.read_csv(
        f"{data_dir}/ratings.dat",
        sep="::",
        names=["user_id", "movie_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )

    movies = pd.read_csv(
        f"{data_dir}/movies.dat",
        sep="::",
        names=["movie_id", "title", "genres"],
        engine="python",
        encoding="latin-1",
    )

    ratings["user_idx"], _ = pd.factorize(ratings["user_id"])
    ratings["movie_idx"], _ = pd.factorize(ratings["movie_id"])

    movie_id_to_idx = dict(zip(ratings["movie_id"], ratings["movie_idx"]))
    movies["movie_idx"] = movies["movie_id"].map(movie_id_to_idx)
    movies = movies.dropna(subset=["movie_idx"])      
    movies["movie_idx"] = movies["movie_idx"].astype(int)

    num_users = ratings["user_idx"].nunique()
    num_movies = ratings["movie_idx"].nunique()

    print(f"Loaded {len(ratings):,} ratings")
    print(f"{num_users:,} users | {num_movies:,} movies")
    print(f"Rating range: {ratings['rating'].min()}{ratings['rating'].max()}")

    train_df, test_df = train_test_split(
        ratings[["user_idx", "movie_idx", "rating"]],
        test_size=0.2,
        random_state=42,
    )

    print(f"Train size: {len(train_df):,} | Test size: {len(test_df):,}")
    return train_df, test_df, movies, num_users, num_movies

if __name__ == "__main__":
    train, test, movies, n_users, n_movies = load_movielens(".")
    print(train.head())
    print(movies.head())