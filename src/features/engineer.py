"""
Feature engineering pipelines and data cleaning functions for YouTube virality modeling.
"""

import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


def clean_videos_and_cohort(
    df_v: pd.DataFrame,
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    df_t: pd.DataFrame,
    seed_file: str,
    start_date: str = "2026-10-11T00:00:00Z",
    end_date: str = "2026-10-17T18:00:00Z",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Clean video corpus and extract a matched cohort with valid observations across 1h, 2h, 4h, 6h."""
    # 1. Filter videos
    mask = (
        (df_v["published_at"] >= start_date)
        & (df_v["published_at"] <= end_date)
        & (df_v["category_id"] != 25)
        & (df_v["live_broadcast_content"] == "none")
        & (df_v["tracked_from_birth"] == True)
    )
    df_v_clean = df_v[mask].copy().sort_values("published_at").reset_index(drop=True)

    # 2. Channel tier mapping from seed_channels.json
    with open(seed_file, "r", encoding="utf-8") as f:
        seeds = json.load(f)

    ch_tier_map = {}
    for tier_name, tier_data in seeds["tiers"].items():
        for ch in tier_data["channels"]:
            ch_tier_map[ch["channel_id"]] = tier_name

    df_v_clean["channel_tier"] = (
        df_v_clean["channel_id"].map(ch_tier_map).fillna("trending_discovered")
    )

    # 3. Ground truth target label
    trending_vids = set(df_t["video_id"])
    df_v_clean["ever_trended"] = df_v_clean["video_id"].isin(trending_vids).astype(int)

    first_trend_df = df_t.groupby("video_id")["captured_at"].min().reset_index()
    first_trend_df.columns = ["video_id", "first_trend_at"]
    df_v_clean = df_v_clean.merge(first_trend_df, on="video_id", how="left")
    df_v_clean["hours_to_trend"] = (
        df_v_clean["first_trend_at"] - df_v_clean["published_at"]
    ).dt.total_seconds() / 3600.0

    # 4. Clean observations
    valid_vids = set(df_v_clean["video_id"])
    df_o_clean = df_o[df_o["video_id"].isin(valid_vids)].copy()
    df_o_clean = df_o_clean.drop_duplicates(
        subset=["video_id", "checkpoint_target_hours"], keep="last"
    )
    df_o_clean = df_o_clean.sort_values(["video_id", "actual_elapsed_hours"])
    df_o_clean["view_count"] = np.maximum(df_o_clean["view_count"], df_o_clean["like_count"])
    df_o_clean["view_count"] = df_o_clean.groupby("video_id")["view_count"].cummax()

    # 5. Clean comments
    df_c_clean = df_c[df_c["video_id"].isin(valid_vids)].copy()
    df_c_clean["elapsed_minutes"] = np.maximum(0.0, df_c_clean["elapsed_minutes"])
    df_c_clean = df_c_clean.drop_duplicates(subset=["video_id", "text"])

    # 6. Checkpoint groupings
    obs_cp1 = df_o_clean[
        (df_o_clean["actual_elapsed_hours"] >= 0.3)
        & (df_o_clean["actual_elapsed_hours"] <= 1.3)
    ]
    obs_cp2 = df_o_clean[
        (df_o_clean["actual_elapsed_hours"] > 1.3)
        & (df_o_clean["actual_elapsed_hours"] <= 2.5)
    ]
    obs_cp4 = df_o_clean[
        (df_o_clean["actual_elapsed_hours"] > 2.5)
        & (df_o_clean["actual_elapsed_hours"] <= 4.5)
    ]
    obs_cp6 = df_o_clean[
        (df_o_clean["actual_elapsed_hours"] > 4.5)
        & (df_o_clean["actual_elapsed_hours"] <= 6.5)
    ]

    common_vids = (
        set(obs_cp1["video_id"])
        .intersection(set(obs_cp2["video_id"]))
        .intersection(set(obs_cp4["video_id"]))
        .intersection(set(obs_cp6["video_id"]))
    )

    df_v_matched = (
        df_v_clean[df_v_clean["video_id"].isin(common_vids)].copy().reset_index(drop=True)
    )

    cp_dict = {
        "cp1": obs_cp1[obs_cp1["video_id"].isin(common_vids)],
        "cp2": obs_cp2[obs_cp2["video_id"].isin(common_vids)],
        "cp4": obs_cp4[obs_cp4["video_id"].isin(common_vids)],
        "cp6": obs_cp6[obs_cp6["video_id"].isin(common_vids)],
    }

    return df_v_matched, df_o_clean, df_c_clean, cp_dict


def extract_static_features(df_v: pd.DataFrame) -> pd.DataFrame:
    """Extract metadata, circadian, and channel rolling upload features."""
    df_feat = pd.DataFrame({"video_id": df_v["video_id"]})

    df_feat["duration_seconds"] = df_v["duration_seconds"].values
    df_feat["is_short"] = (
        (df_v["duration_seconds"] > 0) & (df_v["duration_seconds"] <= 60)
    ).astype(int).values
    df_feat["category_id"] = df_v["category_id"].astype("category").values
    df_feat["channel_tier"] = df_v["channel_tier"].astype("category").values

    titles = df_v["title"].fillna("")
    df_feat["title_len"] = titles.str.len().values
    df_feat["title_word_count"] = titles.apply(lambda s: len(s.split())).values
    df_feat["title_caps_ratio"] = titles.apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(1, len(s))
    ).values
    df_feat["title_has_excl"] = titles.str.contains("!", regex=False).astype(int).values
    df_feat["title_has_quest"] = titles.str.contains("?", regex=False).astype(int).values

    def count_tags(x):
        if pd.isna(x) or x in ("null", "", "[]"):
            return 0
        try:
            return len(json.loads(x))
        except Exception:
            return 0

    df_feat["tag_count"] = df_v["tags"].apply(count_tags).values
    df_feat["definition"] = (df_v["definition"] == "hd").astype(int).values
    df_feat["caption"] = df_v["caption"].astype(int).values
    df_feat["licensed_content"] = df_v["licensed_content"].astype(int).values

    # Circadian encodings
    pub_hours = df_v["published_at"].dt.hour + df_v["published_at"].dt.minute / 60.0
    df_feat["circadian_sin"] = np.sin(2 * np.pi * pub_hours / 24.0).values
    df_feat["circadian_cos"] = np.cos(2 * np.pi * pub_hours / 24.0).values
    pub_dow = df_v["published_at"].dt.dayofweek
    df_feat["dow_sin"] = np.sin(2 * np.pi * pub_dow / 7.0).values
    df_feat["dow_cos"] = np.cos(2 * np.pi * pub_dow / 7.0).values

    # Prior uploads
    df_v_sorted = df_v[["video_id", "channel_id", "published_at"]].copy()
    df_v_sorted["prior_upload_count"] = df_v_sorted.groupby("channel_id").cumcount()
    df_feat["channel_prior_uploads"] = df_v_sorted["prior_upload_count"].values

    return df_feat


def extract_temporal_features(
    df_feat: pd.DataFrame,
    df_c: pd.DataFrame,
    cp_dict: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Extract engagement kinematics and comments features up to 1h, 2h, 4h, and 6h."""
    def get_first_obs(obs_df):
        return (
            obs_df.sort_values("actual_elapsed_hours")
            .groupby("video_id")
            .first()
            .reset_index()
        )

    cp_first = {name: get_first_obs(df) for name, df in cp_dict.items()}

    for name, cp_df in cp_first.items():
        cp_map = cp_df.set_index("video_id")
        df_feat[f"{name}_views"] = (
            df_feat["video_id"].map(cp_map["view_count"]).fillna(0).values
        )
        df_feat[f"{name}_likes"] = (
            df_feat["video_id"].map(cp_map["like_count"]).fillna(0).values
        )
        df_feat[f"{name}_comments"] = (
            df_feat["video_id"].map(cp_map["comment_count"]).fillna(0).values
        )
        df_feat[f"{name}_elapsed_h"] = (
            df_feat["video_id"].map(cp_map["actual_elapsed_hours"]).fillna(1.0).values
        )

    base_cols = [
        "duration_seconds",
        "is_short",
        "category_id",
        "channel_tier",
        "title_len",
        "title_word_count",
        "title_caps_ratio",
        "title_has_excl",
        "title_has_quest",
        "tag_count",
        "definition",
        "caption",
        "licensed_content",
        "circadian_sin",
        "circadian_cos",
        "dow_sin",
        "dow_cos",
        "channel_prior_uploads",
    ]

    # --- Horizon 1h ---
    h1_cols = list(base_cols)
    df_feat["log_views_1h"] = np.log10(df_feat["cp1_views"] + 1)
    df_feat["log_likes_1h"] = np.log10(df_feat["cp1_likes"] + 1)
    df_feat["log_comments_1h"] = np.log10(df_feat["cp1_comments"] + 1)
    df_feat["like_ratio_1h"] = df_feat["cp1_likes"] / (df_feat["cp1_views"] + 1)
    df_feat["comment_ratio_1h"] = df_feat["cp1_comments"] / (df_feat["cp1_views"] + 1)
    df_feat["velocity_1h"] = df_feat["cp1_views"] / np.maximum(
        0.1, df_feat["cp1_elapsed_h"]
    )
    h1_cols += [
        "log_views_1h",
        "log_likes_1h",
        "log_comments_1h",
        "like_ratio_1h",
        "comment_ratio_1h",
        "velocity_1h",
    ]

    c1 = (
        df_c[df_c["elapsed_minutes"] <= 60.0]
        .groupby("video_id")
        .agg(c_count=("comment_id", "count"))
        .reset_index()
        .set_index("video_id")
    )
    df_feat["comments_h1_count"] = (
        df_feat["video_id"].map(c1["c_count"]).fillna(0).values
    )
    df_feat["comments_h1_rate"] = df_feat["comments_h1_count"] / 1.0
    h1_cols += ["comments_h1_count", "comments_h1_rate"]

    # --- Horizon 2h ---
    h2_cols = list(h1_cols)
    df_feat["log_views_2h"] = np.log10(df_feat["cp2_views"] + 1)
    df_feat["log_likes_2h"] = np.log10(df_feat["cp2_likes"] + 1)
    df_feat["like_ratio_2h"] = df_feat["cp2_likes"] / (df_feat["cp2_views"] + 1)
    df_feat["growth_logratio_1_2"] = (
        np.log10(df_feat["cp2_views"] + 1) - np.log10(df_feat["cp1_views"] + 1)
    )

    delta_t_1_2 = np.maximum(0.2, df_feat["cp2_elapsed_h"] - df_feat["cp1_elapsed_h"])
    df_feat["velocity_2h"] = (
        df_feat["cp2_views"] - df_feat["cp1_views"]
    ) / delta_t_1_2
    df_feat["acceleration_1_2"] = (
        df_feat["velocity_2h"] - df_feat["velocity_1h"]
    ) / delta_t_1_2

    c2 = (
        df_c[df_c["elapsed_minutes"] <= 120.0]
        .groupby("video_id")
        .agg(c_count=("comment_id", "count"))
        .reset_index()
        .set_index("video_id")
    )
    df_feat["comments_h2_count"] = (
        df_feat["video_id"].map(c2["c_count"]).fillna(0).values
    )
    df_feat["comments_h2_rate"] = df_feat["comments_h2_count"] / 2.0
    h2_cols += [
        "log_views_2h",
        "log_likes_2h",
        "like_ratio_2h",
        "growth_logratio_1_2",
        "velocity_2h",
        "acceleration_1_2",
        "comments_h2_count",
        "comments_h2_rate",
    ]

    # --- Horizon 4h ---
    h4_cols = list(h2_cols)
    df_feat["log_views_4h"] = np.log10(df_feat["cp4_views"] + 1)
    df_feat["log_likes_4h"] = np.log10(df_feat["cp4_likes"] + 1)
    df_feat["growth_logratio_2_4"] = (
        np.log10(df_feat["cp4_views"] + 1) - np.log10(df_feat["cp2_views"] + 1)
    )

    delta_t_2_4 = np.maximum(0.5, df_feat["cp4_elapsed_h"] - df_feat["cp2_elapsed_h"])
    df_feat["velocity_4h"] = (
        df_feat["cp4_views"] - df_feat["cp2_views"]
    ) / delta_t_2_4
    df_feat["acceleration_2_4"] = (
        df_feat["velocity_4h"] - df_feat["velocity_2h"]
    ) / delta_t_2_4

    c4 = (
        df_c[df_c["elapsed_minutes"] <= 240.0]
        .groupby("video_id")
        .agg(c_count=("comment_id", "count"))
        .reset_index()
        .set_index("video_id")
    )
    df_feat["comments_h4_count"] = (
        df_feat["video_id"].map(c4["c_count"]).fillna(0).values
    )
    df_feat["comments_h4_rate"] = df_feat["comments_h4_count"] / 4.0
    h4_cols += [
        "log_views_4h",
        "log_likes_4h",
        "growth_logratio_2_4",
        "velocity_4h",
        "acceleration_2_4",
        "comments_h4_count",
        "comments_h4_rate",
    ]

    # --- Horizon 6h ---
    h6_cols = list(h4_cols)
    df_feat["log_views_6h"] = np.log10(df_feat["cp6_views"] + 1)
    df_feat["log_likes_6h"] = np.log10(df_feat["cp6_likes"] + 1)
    df_feat["growth_logratio_4_6"] = (
        np.log10(df_feat["cp6_views"] + 1) - np.log10(df_feat["cp4_views"] + 1)
    )

    delta_t_4_6 = np.maximum(0.5, df_feat["cp6_elapsed_h"] - df_feat["cp4_elapsed_h"])
    df_feat["velocity_6h"] = (
        df_feat["cp6_views"] - df_feat["cp4_views"]
    ) / delta_t_4_6
    df_feat["acceleration_4_6"] = (
        df_feat["velocity_6h"] - df_feat["velocity_4h"]
    ) / delta_t_4_6

    c6_s1 = df_c[df_c["elapsed_minutes"] <= 60.0].groupby("video_id").size()
    c6_s2 = (
        df_c[(df_c["elapsed_minutes"] > 60.0) & (df_c["elapsed_minutes"] <= 360.0)]
        .groupby("video_id")
        .size()
    )
    df_feat["comments_s1"] = df_feat["video_id"].map(c6_s1).fillna(0).values
    df_feat["comments_s2"] = df_feat["video_id"].map(c6_s2).fillna(0).values
    df_feat["comment_growth_ratio"] = df_feat["comments_s2"] / (
        df_feat["comments_s1"] + 1
    )
    h6_cols += [
        "log_views_6h",
        "log_likes_6h",
        "growth_logratio_4_6",
        "velocity_6h",
        "acceleration_4_6",
        "comments_s1",
        "comments_s2",
        "comment_growth_ratio",
    ]

    horizons = {"1h": h1_cols, "2h": h2_cols, "4h": h4_cols, "6h": h6_cols}
    return df_feat, horizons


def build_horizon_feature_matrix(
    df_v: pd.DataFrame,
    df_o: pd.DataFrame,
    df_c: pd.DataFrame,
    cp_dict: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Complete feature matrix build step combining static and temporal features."""
    df_feat = extract_static_features(df_v)
    return extract_temporal_features(df_feat, df_c, cp_dict)
