"""
Dual-Stage Comment Dynamics and Sentiment Drift feature extraction.
Extracts temporal sentiment and linguistic reaction drift features across:
  - Stage 1: t <= 1.0h (Subscriber Bubble)
  - Stage 2: 1.0h < t <= 6.0h (Algorithmic Influx)
  - Static 6h: Cumulative reactions up to 6.0h
  - Drift Dynamics: Delta S = S_s2 - S_s1, volume expansion, acceleration, and like shifts.
"""

import os
import time
from multiprocessing import Pool, cpu_count
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def _score_worker(texts: list) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Worker process for vectorized VADER sentiment polarity scoring."""
    analyzer = SentimentIntensityAnalyzer()
    n = len(texts)
    compounds = np.empty(n, dtype=np.float32)
    pos_flags = np.empty(n, dtype=np.int8)
    neg_flags = np.empty(n, dtype=np.int8)
    text_lens = np.empty(n, dtype=np.int16)

    for i, t in enumerate(texts):
        if not t or not isinstance(t, str):
            compounds[i] = 0.0
            pos_flags[i] = 0
            neg_flags[i] = 0
            text_lens[i] = 0
        else:
            text_lens[i] = min(len(t), 32000)
            score = analyzer.polarity_scores(t)
            c = score["compound"]
            compounds[i] = c
            pos_flags[i] = 1 if c >= 0.05 else 0
            neg_flags[i] = 1 if c <= -0.05 else 0

    return compounds, pos_flags, neg_flags, text_lens


def score_comments_vader(df_c: pd.DataFrame, num_workers: Optional[int] = None) -> pd.DataFrame:
    """Score raw comments with VADER sentiment analysis in parallel multiprocessing."""
    if num_workers is None:
        num_workers = min(10, cpu_count())

    texts = df_c["text"].fillna("").tolist()
    total_texts = len(texts)
    batch_size = int(np.ceil(total_texts / max(1, num_workers)))
    batches = [texts[i:i + batch_size] for i in range(0, total_texts, batch_size)]

    print(f"  [Sentiment] Scoring {total_texts:,} comments across {num_workers} worker processes...")
    t_start = time.time()
    with Pool(processes=num_workers) as pool:
        results = pool.map(_score_worker, batches)

    compounds = np.concatenate([r[0] for r in results])
    pos_flags = np.concatenate([r[1] for r in results])
    neg_flags = np.concatenate([r[2] for r in results])
    text_lens = np.concatenate([r[3] for r in results])
    elapsed = time.time() - t_start
    print(f"  [Sentiment] Finished scoring in {elapsed:.2f}s ({total_texts / max(0.01, elapsed):,.0f} comments/sec)")

    df_scored = df_c.copy()
    df_scored["compound"] = compounds
    df_scored["is_pos"] = pos_flags
    df_scored["is_neg"] = neg_flags
    df_scored["text_len"] = text_lens
    return df_scored


def extract_dual_stage_sentiment_features(
    df_v: pd.DataFrame,
    df_c: pd.DataFrame,
    cache_path: Optional[str] = None,
    num_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Extract dual-stage comment dynamics, static sentiment snapshots, and sentiment drift.

    Args:
        df_v: Cleaned video DataFrame (must have 'video_id').
        df_c: Cleaned comments DataFrame.
        cache_path: Optional parquet path to save or reload precomputed features.
        num_workers: Number of CPU workers for parallel sentiment scoring.

    Returns:
        DataFrame aligned with df_v['video_id'] containing all sentiment & drift features.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  [Sentiment] Loading precomputed sentiment features from cache: {cache_path}")
        cached_df = pd.read_parquet(cache_path)
        # Ensure exact alignment with input df_v
        merged = pd.DataFrame({"video_id": df_v["video_id"]}).merge(
            cached_df, on="video_id", how="left"
        )
        return merged

    valid_vids = set(df_v["video_id"])
    df_c_sub = df_c[df_c["video_id"].isin(valid_vids)].copy()
    df_c_sub = df_c_sub[
        (df_c_sub["elapsed_minutes"] >= 0) & (df_c_sub["elapsed_minutes"] <= 360.0)
    ].copy()

    df_scored = score_comments_vader(df_c_sub, num_workers=num_workers)
    df_scored["stage"] = np.where(df_scored["elapsed_minutes"] <= 60.0, 1, 2)

    print("  [Sentiment] Aggregating Stage 1, Stage 2, and cumulative snapshots...")
    grp = df_scored.groupby(["video_id", "stage"]).agg(
        c_mean=("compound", "mean"),
        c_std=("compound", "std"),
        pos_ratio=("is_pos", "mean"),
        neg_ratio=("is_neg", "mean"),
        like_mean=("like_count", "mean"),
        text_len_mean=("text_len", "mean"),
        count=("comment_id", "count"),
    ).unstack(level="stage")

    grp.columns = [f"s{col[1]}_{col[0]}" for col in grp.columns]

    grp_all = df_scored.groupby("video_id").agg(
        s_all_c_mean=("compound", "mean"),
        s_all_c_std=("compound", "std"),
        s_all_pos_ratio=("is_pos", "mean"),
        s_all_neg_ratio=("is_neg", "mean"),
        s_all_like_mean=("like_count", "mean"),
        s_all_text_len_mean=("text_len", "mean"),
        s_all_count=("comment_id", "count"),
    )

    merged = pd.DataFrame({"video_id": df_v["video_id"]})
    merged = merged.merge(grp, on="video_id", how="left").merge(
        grp_all, on="video_id", how="left"
    )

    # Impute missing values for videos with 0 comments in each stage
    for s in [1, 2]:
        merged[f"s{s}_c_mean"] = merged[f"s{s}_c_mean"].fillna(0.0)
        merged[f"s{s}_c_std"] = merged[f"s{s}_c_std"].fillna(0.0)
        merged[f"s{s}_pos_ratio"] = merged[f"s{s}_pos_ratio"].fillna(0.0)
        merged[f"s{s}_neg_ratio"] = merged[f"s{s}_neg_ratio"].fillna(0.0)
        merged[f"s{s}_like_mean"] = merged[f"s{s}_like_mean"].fillna(0.0)
        merged[f"s{s}_text_len_mean"] = merged[f"s{s}_text_len_mean"].fillna(0.0)
        merged[f"s{s}_count"] = merged[f"s{s}_count"].fillna(0)
        merged[f"s{s}_has_comments"] = (merged[f"s{s}_count"] > 0).astype(int)

    merged["s_all_c_mean"] = merged["s_all_c_mean"].fillna(0.0)
    merged["s_all_c_std"] = merged["s_all_c_std"].fillna(0.0)
    merged["s_all_pos_ratio"] = merged["s_all_pos_ratio"].fillna(0.0)
    merged["s_all_neg_ratio"] = merged["s_all_neg_ratio"].fillna(0.0)
    merged["s_all_like_mean"] = merged["s_all_like_mean"].fillna(0.0)
    merged["s_all_text_len_mean"] = merged["s_all_text_len_mean"].fillna(0.0)
    merged["s_all_count"] = merged["s_all_count"].fillna(0)
    merged["s_all_has_comments"] = (merged["s_all_count"] > 0).astype(int)

    # Derive dynamic drift & acceleration features
    merged["sentiment_drift_mean"] = merged["s2_c_mean"] - merged["s1_c_mean"]
    merged["sentiment_pos_drift"] = merged["s2_pos_ratio"] - merged["s1_pos_ratio"]
    merged["sentiment_neg_drift"] = merged["s2_neg_ratio"] - merged["s1_neg_ratio"]
    merged["sentiment_std_drift"] = merged["s2_c_std"] - merged["s1_c_std"]

    # Log volume expansion and comment influx acceleration
    merged["comment_expansion_ratio"] = np.log10(
        (merged["s2_count"] + 1) / (merged["s1_count"] + 1)
    )
    # Rate difference: Stage 2 rate (per hour over 5h) vs Stage 1 rate (over 1h)
    merged["comment_acceleration"] = (merged["s2_count"] / 5.0) - (
        merged["s1_count"] / 1.0
    )
    merged["comment_like_shift"] = np.log10(
        (merged["s2_like_mean"] + 1) / (merged["s1_like_mean"] + 1)
    )
    merged["text_len_drift"] = merged["s2_text_len_mean"] - merged["s1_text_len_mean"]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        merged.to_parquet(cache_path, index=False)
        print(f"  [Sentiment] Cached sentiment features to: {cache_path}")

    return merged
