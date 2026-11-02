#!/usr/bin/env python3
"""
Ablation Experiment 1: Temporal Horizon (The Lead-Time Frontier)
Evaluates model performance across 4 early observation horizons:
  - Horizon 1: t <= 1.0h
  - Horizon 2: t <= 2.0h
  - Horizon 3: t <= 4.0h
  - Horizon 4: t <= 6.0h

Persists all 4 trained model artifacts and evaluation metadata into models/exp1/.
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
from src.models.trainer import ViralModelTrainer


def main():
    start_time = time.time()
    
    # 1. Locate latest Parquet export directory (try data/raw symlink first)
    raw_dir = "/home/kryz/dev/viral/data/raw"
    if not os.path.exists(raw_dir) or not glob.glob(f"{raw_dir}/videos_all_*.parquet"):
        raw_dir = "/home/kryz/dev/harvest/data/export/partition_date=2026-11-02"

    v_file = sorted(glob.glob(os.path.join(raw_dir, "videos_all_*.parquet")))[-1]
    o_file = sorted(glob.glob(os.path.join(raw_dir, "observations_all_*.parquet")))[-1]
    c_file = sorted(glob.glob(os.path.join(raw_dir, "comments_all_*.parquet")))[-1]
    t_file = sorted(glob.glob(os.path.join(raw_dir, "trending_events_all_*.parquet")))[-1]
    seed_file = "/home/kryz/dev/viral/configs/seed_channels.json"

    print("=" * 75)
    print("STEP 1: LOADING & CLEANING MATCHED COHORT")
    print("=" * 75)
    print(f"Data source: {raw_dir}")
    print(f"  Videos:       {os.path.basename(v_file)}")
    print(f"  Observations: {os.path.basename(o_file)}")
    print(f"  Comments:     {os.path.basename(c_file)}")
    print(f"  Trending:     {os.path.basename(t_file)}")

    df_v = pd.read_parquet(v_file)
    df_o = pd.read_parquet(o_file)
    df_c = pd.read_parquet(c_file, columns=['comment_id', 'video_id', 'published_at', 'text', 'like_count', 'elapsed_minutes'])
    df_t = pd.read_parquet(t_file)

    df_v['published_at'] = pd.to_datetime(df_v['published_at'], utc=True)
    df_o['observed_at'] = pd.to_datetime(df_o['observed_at'], utc=True)
    df_c['published_at'] = pd.to_datetime(df_c['published_at'], utc=True)
    df_t['captured_at'] = pd.to_datetime(df_t['captured_at'], utc=True)

    df_v_matched, df_o_clean, df_c_clean, cp_dict = clean_videos_and_cohort(
        df_v, df_o, df_c, df_t, seed_file=seed_file,
        start_date="2026-10-11T00:00:00Z",
        end_date="2026-10-31T00:00:00Z"
    )
    print(f"Matched cohort with valid data at all 4 horizons (1h, 2h, 4h, 6h): {len(df_v_matched):,}")
    print(f"Trending in matched cohort: {df_v_matched['ever_trended'].sum():,} ({df_v_matched['ever_trended'].mean()*100:.2f}%)")

    print("\n" + "=" * 75)
    print("STEP 2: FEATURE ENGINEERING ACROSS HORIZONS")
    print("=" * 75)
    df_feat, horizons = build_horizon_feature_matrix(df_v_matched, df_o_clean, df_c_clean, cp_dict)
    for h_name, cols in horizons.items():
        print(f"  Horizon {h_name:2s}: {len(cols):2d} features")

    # Chronological Split (70% Train, 15% Validation, 15% Test)
    dates = df_v_matched['published_at'].sort_values()
    t_val = dates.quantile(0.70)
    t_test = dates.quantile(0.85)

    train_mask = (df_v_matched['published_at'] < t_val).values
    val_mask = (
        (df_v_matched['published_at'] >= t_val)
        & (df_v_matched['published_at'] < t_test)
    ).values
    test_mask = (df_v_matched['published_at'] >= t_test).values

    print("\n" + "=" * 75)
    print("STEP 3: CHRONOLOGICAL DATA SPLIT")
    print("=" * 75)
    print(f"Train Cohort (< {t_val.strftime('%b %d %H:%M')}):     {np.sum(train_mask):,} videos | Trending: {np.sum(df_v_matched['ever_trended'].values[train_mask]):,} ({np.mean(df_v_matched['ever_trended'].values[train_mask])*100:.2f}%)")
    print(f"Validation Cohort ({t_val.strftime('%b %d')} - {t_test.strftime('%b %d')}): {np.sum(val_mask):,} videos | Trending: {np.sum(df_v_matched['ever_trended'].values[val_mask]):,} ({np.mean(df_v_matched['ever_trended'].values[val_mask])*100:.2f}%)")
    print(f"Test Cohort (>= {t_test.strftime('%b %d %H:%M')}):    {np.sum(test_mask):,} videos | Trending: {np.sum(df_v_matched['ever_trended'].values[test_mask]):,} ({np.mean(df_v_matched['ever_trended'].values[test_mask])*100:.2f}%)")

    print("\n" + "=" * 75)
    print("STEP 4: TRAINING & SERIALIZING MODELS TO models/exp1/")
    print("=" * 75)
    trainer = ViralModelTrainer(output_dir="/home/kryz/dev/viral/models/exp1", random_state=42)

    results = []
    for h_name in ['1h', '2h', '4h', '6h']:
        print(f"\n---> Training & Saving Horizon: t <= {h_name} ...")
        res = trainer.train_horizon(
            horizon_name=h_name,
            feature_cols=horizons[h_name],
            df_feat=df_feat,
            df_v=df_v_matched,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            save_artifact=True
        )
        results.append(res)
        print(f"  PR-AUC:    {res['test_pr_auc']:.4f}  |  ROC-AUC:   {res['test_roc_auc']:.4f}")
        print(f"  F1 Score:  {res['test_f1']:.4f}  |  Precision: {res['test_precision']:.4f}  |  Recall: {res['test_recall']:.4f}")
        print(f"  Saved model artifact to: {res['model_path']}")

    print("\n" + "=" * 85)
    print("FINAL EXPERIMENT 1 RESULTS: THE LEAD-TIME FRONTIER")
    print("=" * 85)
    header = f"{'Horizon':^10} | {'Features':^10} | {'PR-AUC':^10} | {'ROC-AUC':^10} | {'F1':^8} | {'Precision':^10} | {'Recall':^8} | {'Lead Time':^10}"
    print(header)
    print("-" * 85)
    for r in results:
        row = f"{r['horizon']:^10} | {r['features_count']:^10} | {r['test_pr_auc']:^10.4f} | {r['test_roc_auc']:^10.4f} | {r['test_f1']:^8.4f} | {r['test_precision']:^10.4f} | {r['test_recall']:^8.4f} | {r['median_lead_time_hours']:^7.1f}h"
        print(row)
    print("=" * 85)

    report_file = "/home/kryz/dev/viral/reports/exp1_temporal_horizon_results.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'Ablation 1: Temporal Horizon (The Lead-Time Frontier)',
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
