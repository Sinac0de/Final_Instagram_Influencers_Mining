import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RAW_POSTS_JSON = DATA_DIR / "apify_profiles_posts.json"
POSTS_CSV = DATA_DIR / "posts.csv"
USERS_CSV = DATA_DIR / "users.csv"


def load_raw_posts(path: Path) -> pd.DataFrame:
    """Load raw posts JSON from Apify and return as DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    print("Raw columns:", df.columns.tolist())
    print("Number of posts:", len(df))
    return df


def build_posts_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build a clean posts table with one row per post.
    This is the main posts.csv used in the project.
    """
    required_cols = [
        "id",
        "shortCode",
        "type",
        "productType",
        "caption",
        "hashtags",
        "mentions",
        "url",
        "likesCount",
        "commentsCount",
        "timestamp",
        "ownerUsername",
        "ownerFullName",
        "ownerId",
        "isCommentsDisabled",
        "inputUrl",
    ]

    missing = [c for c in required_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Missing expected columns in JSON: {missing}")

    df = df_raw[required_cols].copy()

    # Rename to snake_case
    df.rename(
        columns={
            "id": "ig_post_id",
            "shortCode": "shortcode",
            "type": "post_type",
            "productType": "product_type",
            "url": "post_url",
            "likesCount": "likes",
            "commentsCount": "comments",
            "ownerUsername": "owner_username",
            "ownerFullName": "owner_full_name",
            "ownerId": "owner_id",
            "inputUrl": "owner_profile_url",
        },
        inplace=True,
    )

    # Use shortcode as human‑readable post_id
    df["post_id"] = df["shortcode"].astype(str)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    # Simple raw engagement (later can be normalized per user)
    df["engagement_raw"] = df["likes"] + df["comments"]

    # Sort
    df.sort_values(["owner_username", "timestamp"], ascending=[True, False], inplace=True)

    # Reorder columns
    col_order = [
        "post_id",
        "ig_post_id",
        "shortcode",
        "post_type",
        "product_type",
        "post_url",
        "owner_username",
        "owner_full_name",
        "owner_id",
        "owner_profile_url",
        "caption",
        "hashtags",
        "mentions",
        "likes",
        "comments",
        "engagement_raw",
        "timestamp",
        "isCommentsDisabled",
    ]
    df = df[col_order]

    return df


def build_users_table(df_posts: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate posts table into a users table with summary statistics.
    """
    users = (
        df_posts.groupby(
            ["owner_username", "owner_full_name", "owner_id", "owner_profile_url"],
            dropna=False,
        )
        .agg(
            posts_count=("post_id", "nunique"),
            total_likes=("likes", "sum"),
            total_comments=("comments", "sum"),
            total_engagement=("engagement_raw", "sum"),
            avg_likes=("likes", "mean"),
            avg_comments=("comments", "mean"),
            avg_engagement=("engagement_raw", "mean"),
            first_post_timestamp=("timestamp", "min"),
            last_post_timestamp=("timestamp", "max"),
        )
        .reset_index()
    )

    users.sort_values("total_engagement", ascending=False, inplace=True)

    return users


def main():
    df_raw = load_raw_posts(RAW_POSTS_JSON)
    df_posts = build_posts_table(df_raw)
    df_posts.to_csv(POSTS_CSV, index=False)
    print(f"Saved posts.csv -> {POSTS_CSV.resolve()} (rows: {len(df_posts)})")

    df_users = build_users_table(df_posts)
    df_users.to_csv(USERS_CSV, index=False)
    print(f"Saved users.csv -> {USERS_CSV.resolve()} (rows: {len(df_users)})")


if __name__ == "__main__":
    main()
