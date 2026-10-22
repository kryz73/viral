#!/usr/bin/env python3
"""
Ablation Experiment 2: Feature Modality Contributions (The Multi-Modal Stack)
Fixed at the optimal operating horizon (t <= 4.0h).
Evaluates 4 progressive feature stacks:
  - Stack M1: Metadata Only (16 features)
  - Stack M2: Metadata + Creator Authority Prior (18 features)
  - Stack M3: Metadata + Creator + Engagement Kinematics (35 features)
  - Stack M4: Full Multi-Modal Stack (Above + Comments Signals: 41 features)

Research Question:
  Which data modality drives viral prediction the most: static packaging,
  creator authority, engagement kinematics, or audience comments?
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
    print("EXPERIMENT 2: FEATURE MODALITY CONTRIBUTIONS (THE MULTI-MODAL STACK)")
    print("Operating Horizon: Fixed at t <= 4.0h")
    print("=" * 80)

    # 1. Load and Clean Cohort
    print("\nSTEP 1: LOADING & CLEANING MATCHED COHORT...")
    df_v = pd.read_parquet(v_file)
    df_o = pd.read_parquet(o_file)
    df_c = pd.read_parquet(c_file)
    df_t = pd.read_parquet(t_file)

    df_v['published_at'] = pd.to_datetime(df_v['published_at'], utc=True)
    df_o['observed_at'] = pd.to_datetime(df_o['observed_at'], utc=True)
    df_c['published_at'] = pd.to_datetime(df_c['published_at'], utc=True)
    df_t['captured_at'] = pd.to_datetime(df_t['captured_at'], utc=True)

    df_v_matched, df_o_clean, df_c_clean, cp_dict = clean_videos_and_cohort(
        df_v, df_o, df_c, df_t, seed_file=seed_file
    )
    print(f"Matched cohort size: {len(df_v_matched):,} videos")
    print(f"Trending in cohort:  {df_v_matched['ever_trended'].sum():,} ({df_v_matched['ever_trended'].mean()*100:.2f}%)")

    # 2. Build Feature Matrix
    print("\nSTEP 2: EXTRACTING COMPLETE FEATURE POOL...")
    df_feat, horizons = build_horizon_feature_matrix(df_v_matched, df_o_clean, df_c_clean, cp_dict)

    # Define the 4 Progressive Modality Stacks at t <= 4h:
    meta_cols = [
        "duration_seconds", "is_short", "category_id",
        "title_len", "title_word_count", "title_caps_ratio", "title_has_excl", "title_has_quest",
        "tag_count", "definition", "caption", "licensed_content",
        "circadian_sin", "circadian_cos", "dow_sin", "dow_cos"
    ]

    creator_cols = meta_cols + ["channel_tier", "channel_prior_uploads"]

    kinematics_cols = [
        "log_views_1h", "log_likes_1h", "log_comments_1h", "like_ratio_1h", "comment_ratio_1h", "velocity_1h",
        "log_views_2h", "log_likes_2h", "like_ratio_2h", "growth_logratio_1_2", "velocity_2h", "acceleration_1_2",
        "log_views_4h", "log_likes_4h", "growth_logratio_2_4", "velocity_4h", "acceleration_2_4"
    ]
    m3_cols = creator_cols + kinematics_cols

    comment_cols = [
        "comments_h1_count", "comments_h1_rate",
        "comments_h2_count", "comments_h2_rate",
        "comments_h4_count", "comments_h4_rate"
    ]
    m4_cols = m3_cols + comment_cols

    stacks = {
        "M1_metadata": {
            "name": "M1 (Metadata Only)",
            "cols": meta_cols,
            "description": "Video packaging, title, tags, category, duration, circadian sin/cos"
        },
        "M2_creator": {
            "name": "M2 (+ Creator Prior)",
            "cols": creator_cols,
            "description": "M1 + Channel Tier + Rolling Prior Uploads"
        },
        "M3_kinematics": {
            "name": "M3 (+ Kinematics)",
            "cols": m3_cols,
            "description": "M2 + View/Like Velocities, Accelerations & Growth Ratios up to 4h"
        },
        "M4_comments": {
            "name": "M4 (+ Comments / Full)",
            "cols": m4_cols,
            "description": "M3 + Audience Comment Arrival Dynamics up to 4h"
        }
    }

    for k, v in stacks.items():
        print(f"  {v['name']:25s}: {len(v['cols']):2d} features ({v['description']})")

    # 3. Chronological Splits (Train: Oct 11-15, Val: Oct 16, Test: Oct 17)
    train_mask = (df_v_matched['published_at'] < '2026-10-16T00:00:00Z').values
    val_mask = (
        (df_v_matched['published_at'] >= '2026-10-16T00:00:00Z')
        & (df_v_matched['published_at'] < '2026-10-17T00:00:00Z')
    ).values
    test_mask = (df_v_matched['published_at'] >= '2026-10-17T00:00:00Z').values

    print("\nSTEP 3: CHRONOLOGICAL DATA SPLITS")
    print(f"  Train: {np.sum(train_mask):,} videos ({np.mean(df_v_matched['ever_trended'].values[train_mask])*100:.2f}% positive)")
    print(f"  Val:   {np.sum(val_mask):,} videos ({np.mean(df_v_matched['ever_trended'].values[val_mask])*100:.2f}% positive)")
    print(f"  Test:  {np.sum(test_mask):,} videos ({np.mean(df_v_matched['ever_trended'].values[test_mask])*100:.2f}% positive)")

    # 4. Train and Evaluate each Modality Stack
    print("\nSTEP 4: TRAINING & SERIALIZING MODALITY STACK MODELS TO models/exp2/ ...")
    trainer = ViralModelTrainer(output_dir="/home/kryz/dev/viral/models/exp2", random_state=42)

    results = []
    prev_pr_auc = 0.0
    for stack_id, stack_info in stacks.items():
        print(f"\n---> Training Stack: {stack_info['name']} ({len(stack_info['cols'])} features)...")
        res = trainer.train_horizon(
            horizon_name=stack_id,
            feature_cols=stack_info['cols'],
            df_feat=df_feat,
            df_v=df_v_matched,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            save_artifact=True
        )

        marginal_pr = res['test_pr_auc'] - prev_pr_auc if prev_pr_auc > 0.0 else 0.0
        prev_pr_auc = res['test_pr_auc']
        res['marginal_pr_auc'] = round(marginal_pr, 4)
        res['stack_name'] = stack_info['name']
        res['stack_description'] = stack_info['description']
        results.append(res)

        print(f"  PR-AUC:   {res['test_pr_auc']:.4f} (Δ: {marginal_pr:+.4f}) | ROC-AUC: {res['test_roc_auc']:.4f}")
        print(f"  F1 Score: {res['test_f1']:.4f} | Precision: {res['test_precision']:.4f} | Recall: {res['test_recall']:.4f}")
        print(f"  Saved model to: {res['model_path']}")

    # 5. Output Summary Table
    print("\n" + "=" * 105)
    print("FINAL EXPERIMENT 2 RESULTS: FEATURE MODALITY CONTRIBUTIONS (t <= 4h)")
    print("=" * 105)
    header = f"{'Modality Stack':^25} | {'Features':^8} | {'PR-AUC':^9} | {'Δ PR-AUC':^9} | {'ROC-AUC':^9} | {'F1':^8} | {'Precision':^9} | {'Recall':^8}"
    print(header)
    print("-" * 105)
    for r in results:
        delta_str = f"{r['marginal_pr_auc']:+.4f}" if r['marginal_pr_auc'] != 0.0 else "Baseline"
        row = (
            f"{r['stack_name']:<25} | {r['features_count']:^8} | {r['test_pr_auc']:^9.4f} | "
            f"{delta_str:^9} | {r['test_roc_auc']:^9.4f} | {r['test_f1']:^8.4f} | "
            f"{r['test_precision']:^9.4f} | {r['test_recall']:^8.4f}"
        )
        print(row)
    print("=" * 105)

    # Save to reports
    report_file = "/home/kryz/dev/viral/reports/exp2_feature_modalities_results.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Ablation 2: Feature Modality Contributions (The Multi-Modal Stack)',
            'horizon': 't <= 4.0h',
            'matched_cohort_size': len(df_v_matched),
            'train_size': int(np.sum(train_mask)),
            'val_size': int(np.sum(val_mask)),
            'test_size': int(np.sum(test_mask)),
            'runtime_seconds': round(time.time() - start_time, 2),
            'results': results
        }, f, indent=2)
    print(f"\nDetailed JSON report saved to: {report_file}")


if __name__ == "__main__":
    main()
