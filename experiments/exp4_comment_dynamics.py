#!/usr/bin/env python3
"""
Ablation Experiment 4: Dual-Stage Comment Dynamics & Sentiment Drift (Delta S)
Evaluates how temporal sentiment drift and dynamic audience tone shifts impact virality prediction.

Research Question:
  Does temporal audience sentiment drift (Delta S = S_6h - S_1h) and volume acceleration
  predict breakout virality better than static comment volume or a single static comment snapshot?

Evaluated at the operating horizon t <= 6.0h across 5 controlled setups:
  - Setup C0: Kinematic Baseline (Metadata + Creator Authority + Engagement Physics, ZERO NLP)
  - Setup C1: Baseline + Stage 1 Early Reactions Only (t <= 1.0h, Subscriber Bubble)
  - Setup C2: Baseline + Stage 2 Discovery Reactions Only (1.0h < t <= 6.0h, Algorithmic Influx)
  - Setup C_static: Baseline + Static 6h Cumulative Snapshot (Flat sentiment across all comments <= 6h)
  - Setup C3: Dual-Stage Drift Ensemble (Stage 1 + Stage 2 + Sentiment Drift Delta S + Influx Dynamics)

Saves all trained LightGBM models into models/exp4/ and outputs reports/exp4_comment_dynamics_results.json.
"""

import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd

# Add repo root to pythonpath
sys.path.insert(0, "/home/kryz/dev/viral")

from src.features.engineer import clean_videos_and_cohort, build_horizon_feature_matrix
from src.features.sentiment import extract_dual_stage_sentiment_features
from src.models.trainer import ViralModelTrainer


def main():
    start_time = time.time()

    # 1. Locate latest Parquet export directory
    raw_dir = "/home/kryz/dev/viral/data/raw"
    if not os.path.exists(raw_dir) or not glob.glob(f"{raw_dir}/videos_all_*.parquet"):
        raw_dir = "/home/kryz/dev/harvest/data/export/partition_date=2026-11-02"

    v_file = sorted(glob.glob(os.path.join(raw_dir, "videos_all_*.parquet")))[-1]
    o_file = sorted(glob.glob(os.path.join(raw_dir, "observations_all_*.parquet")))[-1]
    c_file = sorted(glob.glob(os.path.join(raw_dir, "comments_all_*.parquet")))[-1]
    t_file = sorted(glob.glob(os.path.join(raw_dir, "trending_events_all_*.parquet")))[-1]
    seed_file = "/home/kryz/dev/viral/configs/seed_channels.json"

    print("=" * 80)
    print("EXPERIMENT 4: DUAL-STAGE COMMENT DYNAMICS & SENTIMENT DRIFT")
    print("Operating Horizon: t <= 6.0h | Stage 1 (t <= 1h) vs Stage 2 (1h < t <= 6h)")
    print("=" * 80)
    print(f"Data source: {raw_dir}")
    print(f"  Videos:       {os.path.basename(v_file)}")
    print(f"  Observations: {os.path.basename(o_file)}")
    print(f"  Comments:     {os.path.basename(c_file)}")
    print(f"  Trending:     {os.path.basename(t_file)}")

    # 1. Load and Clean Matched Cohort
    print("\nSTEP 1: LOADING & CLEANING MATCHED COHORT...")
    df_v = pd.read_parquet(v_file)
    df_o = pd.read_parquet(o_file)
    df_c = pd.read_parquet(
        c_file,
        columns=["comment_id", "video_id", "published_at", "text", "like_count", "elapsed_minutes"],
    )
    df_t = pd.read_parquet(t_file)

    df_v["published_at"] = pd.to_datetime(df_v["published_at"], utc=True)
    df_o["observed_at"] = pd.to_datetime(df_o["observed_at"], utc=True)
    df_c["published_at"] = pd.to_datetime(df_c["published_at"], utc=True)
    df_t["captured_at"] = pd.to_datetime(df_t["captured_at"], utc=True)

    df_v_matched, df_o_clean, df_c_clean, cp_dict = clean_videos_and_cohort(
        df_v, df_o, df_c, df_t, seed_file=seed_file,
        start_date="2026-10-11T00:00:00Z",
        end_date="2026-10-31T00:00:00Z"
    )
    print(f"Matched cohort size: {len(df_v_matched):,} videos")
    print(f"Trending in cohort:  {df_v_matched['ever_trended'].sum():,} ({df_v_matched['ever_trended'].mean()*100:.2f}%)")

    # 2. Extract Base Horizon Features and Kinematics up to 6h
    print("\nSTEP 2: EXTRACTING HORIZON 6H KINEMATICS & BASE FEATURES...")
    df_feat, horizons = build_horizon_feature_matrix(df_v_matched, df_o_clean, df_c_clean, cp_dict)

    # Calculate additional physical kinematic derivatives up to 6h
    delta_t_1_2 = np.maximum(0.2, df_feat["cp2_elapsed_h"] - df_feat["cp1_elapsed_h"])
    delta_t_2_4 = np.maximum(0.5, df_feat["cp4_elapsed_h"] - df_feat["cp2_elapsed_h"])
    delta_t_4_6 = np.maximum(0.5, df_feat["cp6_elapsed_h"] - df_feat["cp4_elapsed_h"])

    df_feat["like_velocity_1h"] = df_feat["cp1_likes"] / np.maximum(0.1, df_feat["cp1_elapsed_h"])
    df_feat["like_velocity_2h"] = (df_feat["cp2_likes"] - df_feat["cp1_likes"]) / delta_t_1_2
    df_feat["like_velocity_4h"] = (df_feat["cp4_likes"] - df_feat["cp2_likes"]) / delta_t_2_4
    df_feat["like_velocity_6h"] = (df_feat["cp6_likes"] - df_feat["cp4_likes"]) / delta_t_4_6

    df_feat["like_acceleration_1_2"] = (df_feat["like_velocity_2h"] - df_feat["like_velocity_1h"]) / delta_t_1_2
    df_feat["like_acceleration_2_4"] = (df_feat["like_velocity_4h"] - df_feat["like_velocity_2h"]) / delta_t_2_4
    df_feat["like_acceleration_4_6"] = (df_feat["like_velocity_6h"] - df_feat["like_velocity_4h"]) / delta_t_4_6

    df_feat["like_growth_logratio_1_2"] = np.log10(df_feat["cp2_likes"] + 1) - np.log10(df_feat["cp1_likes"] + 1)
    df_feat["like_growth_logratio_2_4"] = np.log10(df_feat["cp4_likes"] + 1) - np.log10(df_feat["cp2_likes"] + 1)
    df_feat["like_growth_logratio_4_6"] = np.log10(df_feat["cp6_likes"] + 1) - np.log10(df_feat["cp4_likes"] + 1)

    df_feat["like_ratio_6h"] = df_feat["cp6_likes"] / (df_feat["cp6_views"] + 1)
    df_feat["comment_ratio_6h"] = df_feat["cp6_comments"] / (df_feat["cp6_views"] + 1)

    # 3. Extract or Load Dual-Stage Sentiment & Drift Features
    print("\nSTEP 3: COMPUTING DUAL-STAGE SENTIMENT & DRIFT (VADER NLP)...")
    cache_path = "/home/kryz/dev/viral/data/processed/video_comment_sentiment_features.parquet"
    df_sent = extract_dual_stage_sentiment_features(
        df_v_matched, df_c_clean, cache_path=cache_path
    )

    # Merge sentiment features into feature matrix (avoiding fragmentation)
    sent_cols_to_add = [c for c in df_sent.columns if c != "video_id"]
    df_feat = pd.concat([df_feat, df_sent[sent_cols_to_add]], axis=1)

    # Base metadata + Creator authority
    base_cols = [
        "duration_seconds", "is_short", "category_id", "channel_tier",
        "title_len", "title_word_count", "title_caps_ratio", "title_has_excl",
        "title_has_quest", "tag_count", "definition", "caption",
        "licensed_content", "circadian_sin", "circadian_cos", "dow_sin",
        "dow_cos", "channel_prior_uploads"
    ]

    # Full Kinematics at 6h (No NLP text sentiment)
    c0_cols = base_cols + [
        # Cumulative counts
        "log_views_1h", "log_likes_1h", "log_comments_1h",
        "log_views_2h", "log_likes_2h",
        "log_views_4h", "log_likes_4h",
        "log_views_6h", "log_likes_6h",
        # Derivatives & Log-ratios
        "velocity_1h", "velocity_2h", "velocity_4h", "velocity_6h",
        "like_velocity_1h", "like_velocity_2h", "like_velocity_4h", "like_velocity_6h",
        "acceleration_1_2", "acceleration_2_4", "acceleration_4_6",
        "like_acceleration_1_2", "like_acceleration_2_4", "like_acceleration_4_6",
        "growth_logratio_1_2", "growth_logratio_2_4", "growth_logratio_4_6",
        "like_growth_logratio_1_2", "like_growth_logratio_2_4", "like_growth_logratio_4_6",
        # Interaction ratios & raw counts
        "like_ratio_1h", "like_ratio_2h", "like_ratio_6h",
        "comment_ratio_1h", "comment_ratio_6h",
        "comments_s1", "comments_s2", "comment_growth_ratio"
    ]

    # Stage 1 Sentiment Features (Subscriber Bubble, t <= 1h)
    s1_features = [
        "s1_c_mean", "s1_c_std", "s1_pos_ratio", "s1_neg_ratio",
        "s1_like_mean", "s1_text_len_mean", "s1_has_comments"
    ]

    # Stage 2 Sentiment Features (Algorithmic Influx, 1h < t <= 6h)
    s2_features = [
        "s2_c_mean", "s2_c_std", "s2_pos_ratio", "s2_neg_ratio",
        "s2_like_mean", "s2_text_len_mean", "s2_has_comments"
    ]

    # Static 6h Cumulative Sentiment Features (Flat across all comments <= 6h)
    static_features = [
        "s_all_c_mean", "s_all_c_std", "s_all_pos_ratio", "s_all_neg_ratio",
        "s_all_like_mean", "s_all_text_len_mean", "s_all_has_comments"
    ]

    # Dynamic Drift & Temporal Influx Features
    drift_features = [
        "sentiment_drift_mean", "sentiment_pos_drift", "sentiment_neg_drift",
        "sentiment_std_drift", "comment_expansion_ratio", "comment_acceleration",
        "comment_like_shift", "text_len_drift"
    ]

    # Define the 5 Experimental Setups
    c1_cols = c0_cols + s1_features
    c2_cols = c0_cols + s2_features
    c_static_cols = c0_cols + static_features
    c3_cols = c0_cols + s1_features + s2_features + drift_features

    setups = {
        "C0_kinematic_baseline": {
            "name": "C0 (Kinematic Baseline)",
            "cols": c0_cols,
            "description": "Full tabular kinematics up to 6h (ZERO text/sentiment features)"
        },
        "C1_stage1_early": {
            "name": "C1 (+ Stage 1 Early Reactions)",
            "cols": c1_cols,
            "description": "C0 + Early subscriber sentiment & reaction tone at t <= 1.0h"
        },
        "C2_stage2_discovery": {
            "name": "C2 (+ Stage 2 Influx Reactions)",
            "cols": c2_cols,
            "description": "C0 + Algorithmic discovery reaction sentiment at 1.0h < t <= 6.0h"
        },
        "C_static_6h_snapshot": {
            "name": "C_static (+ Static 6h Snapshot)",
            "cols": c_static_cols,
            "description": "C0 + Cumulative flat sentiment snapshot over all comments up to 6h"
        },
        "C3_dual_stage_drift": {
            "name": "C3 (Dual-Stage Drift Ensemble)",
            "cols": c3_cols,
            "description": "C0 + S1 + S2 + Temporal Drift (Delta S = S2 - S1) + Influx Acceleration"
        }
    }

    print("\nEXPERIMENTAL SETUPS CONFIGURED:")
    for k, v in setups.items():
        print(f"  {v['name']:35s}: {len(v['cols']):2d} features ({v['description']})")

    # 4. Chronological Splits (70% Train, 15% Validation, 15% Test)
    dates = df_v_matched["published_at"].sort_values()
    t_val = dates.quantile(0.70)
    t_test = dates.quantile(0.85)

    train_mask = (df_v_matched["published_at"] < t_val).values
    val_mask = (
        (df_v_matched["published_at"] >= t_val)
        & (df_v_matched["published_at"] < t_test)
    ).values
    test_mask = (df_v_matched["published_at"] >= t_test).values

    y = df_v_matched["ever_trended"].values.astype(int)

    print(f"\nChronological Splits:")
    print(f"  Train:      {train_mask.sum():,} videos (< {t_val.strftime('%Y-%m-%d %H:%M')}) | Positives: {y[train_mask].sum():,} ({y[train_mask].mean()*100:.2f}%)")
    print(f"  Validation: {val_mask.sum():,} videos ({t_val.strftime('%Y-%m-%d %H:%M')} to {t_test.strftime('%Y-%m-%d %H:%M')}) | Positives: {y[val_mask].sum():,} ({y[val_mask].mean()*100:.2f}%)")
    print(f"  Test:       {test_mask.sum():,} videos (>= {t_test.strftime('%Y-%m-%d %H:%M')}) | Positives: {y[test_mask].sum():,} ({y[test_mask].mean()*100:.2f}%)")

    # 5. Train and Serialize Models
    output_dir = "/home/kryz/dev/viral/models/exp4"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSTEP 4: TRAINING & SERIALIZING LIGHTGBM BOOSTERS TO {output_dir} ...")
    trainer = ViralModelTrainer(output_dir=output_dir, random_state=42)

    results = []
    base_pr_auc = None

    for sid, sinfo in setups.items():
        print(f"\n---> Training Setup: {sinfo['name']} ({len(sinfo['cols'])} features)...")
        res = trainer.train_horizon(
            horizon_name=sid,
            feature_cols=sinfo["cols"],
            df_feat=df_feat,
            df_v=df_v_matched,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            save_artifact=True
        )

        if base_pr_auc is None:
            base_pr_auc = res["test_pr_auc"]
            marginal_lift = 0.0
        else:
            marginal_lift = round(res["test_pr_auc"] - base_pr_auc, 4)

        res["setup_id"] = sid
        res["setup_name"] = sinfo["name"]
        res["setup_description"] = sinfo["description"]
        res["marginal_pr_auc_vs_c0"] = marginal_lift
        results.append(res)

        print(f"  PR-AUC:    {res['test_pr_auc']:.4f} (Delta vs C0: {marginal_lift:+.4f}) | ROC-AUC: {res['test_roc_auc']:.4f}")
        print(f"  F1 Score:  {res['test_f1']:.4f} | Precision: {res['test_precision']:.2%} | Recall: {res['test_recall']:.2%}")
        print(f"  Saved booster: {res['model_path']}")

    # 6. Final Comparative Output Table
    print("\n" + "=" * 105)
    print("FINAL EXPERIMENT 4 RESULTS: DUAL-STAGE COMMENT DYNAMICS & SENTIMENT DRIFT")
    print("=" * 105)
    header = f"{'Setup':<35} | {'Feats':^6} | {'PR-AUC':^9} | {'Delta PR':^9} | {'ROC-AUC':^9} | {'F1':^8} | {'Prec':^8} | {'Recall':^8}"
    print(header)
    print("-" * 105)
    for r in results:
        row = (
            f"{r['setup_name']:<35} | {r['features_count']:^6d} | "
            f"{r['test_pr_auc']:^9.4f} | {r['marginal_pr_auc_vs_c0']:^+9.4f} | "
            f"{r['test_roc_auc']:^9.4f} | {r['test_f1']:^8.4f} | "
            f"{r['test_precision']:^7.1%} | {r['test_recall']:^7.1%}"
        )
        print(row)
    print("=" * 105)

    # 7. Save JSON Report
    reports_dir = "/home/kryz/dev/viral/reports"
    os.makedirs(reports_dir, exist_ok=True)
    report_file = os.path.join(reports_dir, "exp4_comment_dynamics_results.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Ablation 4: Dual-Stage Comment Dynamics & Sentiment Drift (Delta S)",
            "operating_horizon": "t <= 6.0h",
            "stages": {
                "stage_1": "t <= 1.0h (Subscriber Bubble)",
                "stage_2": "1.0h < t <= 6.0h (Algorithmic Influx)",
                "static_6h": "t <= 6.0h (Cumulative Snapshot)"
            },
            "matched_cohort_size": len(df_v_matched),
            "train_size": int(np.sum(train_mask)),
            "val_size": int(np.sum(val_mask)),
            "test_size": int(np.sum(test_mask)),
            "runtime_seconds": round(time.time() - start_time, 2),
            "results": results
        }, f, indent=2)

    print(f"\nDetailed JSON report saved to: {report_file}")
    print(f"Experiment 4 completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
