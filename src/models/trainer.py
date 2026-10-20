"""
ViralModelTrainer handles training, early stopping, validation threshold tuning,
and serializing trained models to disk.
"""

import json
import os
from typing import Any, Dict, List, Optional
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


class ViralModelTrainer:
    def __init__(
        self,
        output_dir: str = "models",
        random_state: int = 42,
    ):
        self.output_dir = output_dir
        self.random_state = random_state
        os.makedirs(self.output_dir, exist_ok=True)

    def train_horizon(
        self,
        horizon_name: str,
        feature_cols: List[str],
        df_feat: pd.DataFrame,
        df_v: pd.DataFrame,
        train_mask: np.ndarray,
        val_mask: np.ndarray,
        test_mask: np.ndarray,
        save_artifact: bool = True,
    ) -> Dict[str, Any]:
        """Train LightGBM on train split, tune threshold on val split, evaluate on test split, and save model."""
        X = df_feat[feature_cols].copy()
        y = df_v["ever_trended"].values

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = float(neg_count / max(1, pos_count))

        clf = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

        clf.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
        )

        # Predict probabilities
        val_probs = clf.predict_proba(X_val)[:, 1]
        test_probs = clf.predict_proba(X_test)[:, 1]

        # Optimal F1 threshold from validation set
        precisions, recalls, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * (precisions * recalls) / (precisions + recalls),
            0,
        )
        best_idx = np.argmax(f1_scores)
        best_threshold = (
            float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        )

        # Test set evaluation
        test_preds = (test_probs >= best_threshold).astype(int)
        test_roc_auc = float(roc_auc_score(y_test, test_probs))
        test_pr_auc = float(average_precision_score(y_test, test_probs))
        test_f1 = float(f1_score(y_test, test_preds, zero_division=0))
        test_precision = float(precision_score(y_test, test_preds, zero_division=0))
        test_recall = float(recall_score(y_test, test_preds, zero_division=0))
        test_brier = float(brier_score_loss(y_test, test_probs))

        # Median lead time for trending test samples
        test_v = df_v[test_mask]
        test_trend_hours = test_v[test_v["ever_trended"] == 1][
            "hours_to_trend"
        ].dropna()
        median_lead_time = (
            float(test_trend_hours.median()) if len(test_trend_hours) > 0 else 0.0
        )

        imp = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(
            ascending=False
        )
        top_features = imp.head(10).to_dict()

        result = {
            "horizon": horizon_name,
            "features_count": len(feature_cols),
            "best_iteration": int(clf.best_iteration_),
            "optimal_threshold": round(best_threshold, 4),
            "test_pr_auc": round(test_pr_auc, 4),
            "test_roc_auc": round(test_roc_auc, 4),
            "test_f1": round(test_f1, 4),
            "test_precision": round(test_precision, 4),
            "test_recall": round(test_recall, 4),
            "test_brier": round(test_brier, 4),
            "median_lead_time_hours": round(median_lead_time, 1),
            "top_features": top_features,
            "feature_names": feature_cols,
        }

        # Save model artifact to disk
        if save_artifact:
            model_path = os.path.join(self.output_dir, f"lgbm_{horizon_name}.txt")
            clf.booster_.save_model(model_path)
            meta_path = os.path.join(self.output_dir, f"metadata_{horizon_name}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            result["model_path"] = model_path
            result["meta_path"] = meta_path

        return result
