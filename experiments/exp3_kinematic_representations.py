#!/usr/bin/env python3
"""
Ablation Experiment 3: Kinematic Representations (Engagement Physics)
Evaluates how different physical formulations of engagement dynamics impact virality prediction.

Research Question:
  Does modeling engagement as continuous derivatives (velocity v = dV/dt, acceleration a = d^2V/dt^2)
  and scale-invariant multiplicative expansion (rho = Delta log V) outperform raw cumulative counts?

Evaluated at the fixed operating horizon t <= 4.0h across 6 controlled formulations:
  - Setup K1: Cumulative Counts Only (0th Derivative - Scale Only)
  - Setup K2: Counts + 1st Derivative (Velocity v - Influx Rate)
  - Setup K3: Counts + Velocity + 2nd Derivative (Acceleration a - Curvature)
  - Setup K4: Counts + Velocity + Acceleration + Relative Multipliers (Growth Log-Ratios rho)
  - Setup K5: Full Physics Stack (Above + Cross-Modality Interaction Ratios)
  - Setup K0: Scale-Free Kinematics (Pure Dynamics - Zero Cumulative Counts)

Saves all trained LightGBM models into models/exp3/ and outputs reports/exp3_kinematic_representations_results.json.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

# Add repo root to pythonpath
sys.path.insert(0, "/home/kryz/dev/viral")

from src.features.engineer import clean_videos_and_cohort, build_horizon_feature_matrix
from src.models.trainer import ViralModelTrainer


def main():
    start_time = time.time()
    data_dir = "/home/kryz/dev/harvest/data/export/partition_date=2026-10-20"
    v_file = os.path.join(data_dir, "videos_all_20261020_095610.parquet")
    o_file = os.path.join(data_dir, "observations_all_20261020_095610.parquet")
    c_file = os.path.join(data_dir, "comments_all_20261020_095610.parquet")
    t_file = os.path.join(data_dir, "trending_events_all_20261020_095610.parquet")
    seed_file = "/home/kryz/dev/viral/configs/seed_channels.json"

    print("=" * 80)
    print("EXPERIMENT 3: KINEMATIC REPRESENTATIONS (ENGAGEMENT PHYSICS)")
    print("Operating Horizon: Fixed at t <= 4.0h | Matched Clean Cohort")
    print("=" * 80)

    # 1. Load and Clean Matched Cohort
    print("\nSTEP 1: LOADING & CLEANING MATCHED COHORT...")
    df_v = pd.read_parquet(v_file)
    df_o = pd.read_parquet(o_file)
    df_c = pd.read_parquet(c_file)
    df_t = pd.read_parquet(t_file)

    df_v["published_at"] = pd.to_datetime(df_v["published_at"], utc=True)
    df_o["observed_at"] = pd.to_datetime(df_o["observed_at"], utc=True)
    df_c["published_at"] = pd.to_datetime(df_c["published_at"], utc=True)
    df_t["captured_at"] = pd.to_datetime(df_t["captured_at"], utc=True)

    df_v_matched, df_o_clean, df_c_clean, cp_dict = clean_videos_and_cohort(
        df_v, df_o, df_c, df_t, seed_file=seed_file
    )
    print(f"Matched cohort size: {len(df_v_matched):,} videos")
    print(f"Trending in cohort:  {df_v_matched['ever_trended'].sum():,} ({df_v_matched['ever_trended'].mean()*100:.2f}%)")

    # 2. Extract Base Horizon Features and Derive Complete Kinematic Physics
    print("\nSTEP 2: COMPUTING PHYSICAL DERIVATIVES & KINEMATIC FEATURES...")
    df_feat, _ = build_horizon_feature_matrix(df_v_matched, df_o_clean, df_c_clean, cp_dict)

    # Interval durations
    delta_t_1_2 = np.maximum(0.2, df_feat["cp2_elapsed_h"] - df_feat["cp1_elapsed_h"])
    delta_t_2_4 = np.maximum(0.5, df_feat["cp4_elapsed_h"] - df_feat["cp2_elapsed_h"])

    # Comments snapshots & interaction ratios
    df_feat["log_comments_2h"] = np.log10(df_feat["cp2_comments"] + 1)
    df_feat["log_comments_4h"] = np.log10(df_feat["cp4_comments"] + 1)
    df_feat["comment_ratio_4h"] = df_feat["cp4_comments"] / (df_feat["cp4_views"] + 1)

    # Like Kinematics (Velocities, Accelerations, Log-Ratios)
    df_feat["like_velocity_1h"] = df_feat["cp1_likes"] / np.maximum(0.1, df_feat["cp1_elapsed_h"])
    df_feat["like_velocity_2h"] = (df_feat["cp2_likes"] - df_feat["cp1_likes"]) / delta_t_1_2
    df_feat["like_velocity_4h"] = (df_feat["cp4_likes"] - df_feat["cp2_likes"]) / delta_t_2_4

    df_feat["like_acceleration_1_2"] = (df_feat["like_velocity_2h"] - df_feat["like_velocity_1h"]) / delta_t_1_2
    df_feat["like_acceleration_2_4"] = (df_feat["like_velocity_4h"] - df_feat["like_velocity_2h"]) / delta_t_2_4

    df_feat["like_growth_logratio_1_2"] = np.log10(df_feat["cp2_likes"] + 1) - np.log10(df_feat["cp1_likes"] + 1)
    df_feat["like_growth_logratio_2_4"] = np.log10(df_feat["cp4_likes"] + 1) - np.log10(df_feat["cp2_likes"] + 1)
    df_feat["like_ratio_4h"] = df_feat["cp4_likes"] / (df_feat["cp4_views"] + 1)

    # Base metadata + Creator authority (Controlled across all formulations)
    base_cols = [
        "duration_seconds", "is_short", "category_id", "channel_tier",
        "title_len", "title_word_count", "title_caps_ratio", "title_has_excl",
        "title_has_quest", "tag_count", "definition", "caption",
        "licensed_content", "circadian_sin", "circadian_cos", "dow_sin",
        "dow_cos", "channel_prior_uploads"
    ]

    # Setup K1: Pure Cumulative Counts (0th Order - Scale Only)
    k1_cols = base_cols + [
        "log_views_1h", "log_likes_1h", "log_comments_1h",
        "log_views_2h", "log_likes_2h", "log_comments_2h",
        "log_views_4h", "log_likes_4h", "log_comments_4h"
    ]

    # Setup K2: Counts + 1st Derivative (Velocity v - Influx Rate)
    k2_cols = k1_cols + [
        "velocity_1h", "velocity_2h", "velocity_4h",
        "like_velocity_1h", "like_velocity_2h", "like_velocity_4h"
    ]

    # Setup K3: Counts + Velocity + 2nd Derivative (Acceleration a - Curvature)
    k3_cols = k2_cols + [
        "acceleration_1_2", "acceleration_2_4",
        "like_acceleration_1_2", "like_acceleration_2_4"
    ]

    # Setup K4: Counts + Velocity + Acceleration + Relative Multipliers (Growth Log-Ratios rho)
    k4_cols = k3_cols + [
        "growth_logratio_1_2", "growth_logratio_2_4",
        "like_growth_logratio_1_2", "like_growth_logratio_2_4"
    ]

    # Setup K5: Full Integrated Physics (+ Cross-Modality Interaction Ratios)
    k5_cols = k4_cols + [
        "like_ratio_1h", "like_ratio_2h", "like_ratio_4h",
        "comment_ratio_1h", "comment_ratio_4h"
    ]

    # Setup K0: Scale-Free Differential Kinematics (No Cumulative Counts)
    k0_cols = base_cols + [
        "velocity_1h", "velocity_2h", "velocity_4h",
        "like_velocity_1h", "like_velocity_2h", "like_velocity_4h",
        "acceleration_1_2", "acceleration_2_4",
        "like_acceleration_1_2", "like_acceleration_2_4",
        "growth_logratio_1_2", "growth_logratio_2_4",
        "like_growth_logratio_1_2", "like_growth_logratio_2_4",
        "like_ratio_1h", "like_ratio_2h", "like_ratio_4h",
        "comment_ratio_1h", "comment_ratio_4h"
    ]

    formulations = {
        "K1_counts": {
            "name": "K1 (Cumulative Counts)",
            "cols": k1_cols,
            "description": "0th-order cumulative log counts of views, likes, and comments up to 4h"
        },
        "K2_velocity": {
            "name": "K2 (+ Velocity)",
            "cols": k2_cols,
            "description": "K1 + 1st derivatives: instantaneous view & like velocity across intervals"
        },
        "K3_acceleration": {
            "name": "K3 (+ Acceleration)",
            "cols": k3_cols,
            "description": "K2 + 2nd derivatives: view & like acceleration (curvature / ignition)"
        },
        "K4_logratios": {
            "name": "K4 (+ Growth Log-Ratios)",
            "cols": k4_cols,
            "description": "K3 + Multiplicative scale-invariant log growth ratios (rho = Delta log V)"
        },
        "K5_full_physics": {
            "name": "K5 (Full Physics)",
            "cols": k5_cols,
            "description": "K4 + Cross-modality engagement interaction density (like & comment ratios)"
        },
        "K0_scale_free": {
            "name": "K0 (Scale-Free Physics)",
            "cols": k0_cols,
            "description": "Pure kinematic dynamics (v, a, rho, ratios) with ZERO cumulative view/like counts"
        }
    }

    for k, v in formulations.items():
        print(f"  {v['name']:25s}: {len(v['cols']):2d} features ({v['description']})")

    # 3. Chronological Splits (Train: Oct 11-15, Val: Oct 16, Test: Oct 17)
    train_mask = (df_v_matched["published_at"] < "2026-10-16T00:00:00Z").values
    val_mask = (
        (df_v_matched["published_at"] >= "2026-10-16T00:00:00Z")
        & (df_v_matched["published_at"] < "2026-10-17T00:00:00Z")
    ).values
    test_mask = (df_v_matched["published_at"] >= "2026-10-17T00:00:00Z").values

    print("\nSTEP 3: CHRONOLOGICAL DATA SPLITS")
    print(f"  Train: {np.sum(train_mask):,} videos ({np.mean(df_v_matched['ever_trended'].values[train_mask])*100:.2f}% positive)")
    print(f"  Val:   {np.sum(val_mask):,} videos ({np.mean(df_v_matched['ever_trended'].values[val_mask])*100:.2f}% positive)")
    print(f"  Test:  {np.sum(test_mask):,} videos ({np.mean(df_v_matched['ever_trended'].values[test_mask])*100:.2f}% positive)")

    # 4. Train and Serialize Models
    output_dir = "/home/kryz/dev/viral/models/exp3"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSTEP 4: TRAINING & SERIALIZING KINEMATIC BOOSTERS TO {output_dir} ...")
    trainer = ViralModelTrainer(output_dir=output_dir, random_state=42)

    results = []
    base_pr_auc = None

    for fid, finfo in formulations.items():
        print(f"\n---> Training Formulation: {finfo['name']} ({len(finfo['cols'])} features)...")
        res = trainer.train_horizon(
            horizon_name=fid,
            feature_cols=finfo["cols"],
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

        res["formulation_id"] = fid
        res["formulation_name"] = finfo["name"]
        res["formulation_description"] = finfo["description"]
        res["marginal_pr_auc_vs_k1"] = marginal_lift
        results.append(res)

        print(f"  PR-AUC:    {res['test_pr_auc']:.4f} (Delta vs K1: {marginal_lift:+.4f}) | ROC-AUC: {res['test_roc_auc']:.4f}")
        print(f"  F1 Score:  {res['test_f1']:.4f} | Precision: {res['test_precision']:.2%} | Recall: {res['test_recall']:.2%}")
        print(f"  Saved booster: {res['model_path']}")

    # 5. Final Output Table
    print("\n" + "=" * 96)
    print("FINAL EXPERIMENT 3 RESULTS: KINEMATIC REPRESENTATIONS (ENGAGEMENT PHYSICS)")
    print("=" * 96)
    header = f"{'Formulation':^23} | {'Feats':^6} | {'PR-AUC':^9} | {'Delta PR':^9} | {'ROC-AUC':^9} | {'F1':^8} | {'Prec':^8} | {'Recall':^8}"
    print(header)
    print("-" * 96)
    for r in results:
        row = (
            f"{r['formulation_name']:<23} | {r['features_count']:^6d} | "
            f"{r['test_pr_auc']:^9.4f} | {r['marginal_pr_auc_vs_k1']:^+9.4f} | "
            f"{r['test_roc_auc']:^9.4f} | {r['test_f1']:^8.4f} | "
            f"{r['test_precision']:^7.1%} | {r['test_recall']:^7.1%}"
        )
        print(row)
    print("=" * 96)

    # 6. Save JSON Report
    report_file = "/home/kryz/dev/viral/reports/exp3_kinematic_representations_results.json"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "Ablation 3: Kinematic Representations (Engagement Physics)",
            "operating_horizon": "t <= 4.0h",
            "matched_cohort_size": len(df_v_matched),
            "train_size": int(np.sum(train_mask)),
            "val_size": int(np.sum(val_mask)),
            "test_size": int(np.sum(test_mask)),
            "runtime_seconds": round(time.time() - start_time, 2),
            "results": results
        }, f, indent=2)

    print(f"\nDetailed JSON report saved to: {report_file}")
    print(f"Experiment 3 completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
