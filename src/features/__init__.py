"""
Feature engineering pipelines for YouTube viral prediction.
"""

from .engineer import (
    clean_videos_and_cohort,
    extract_static_features,
    extract_temporal_features,
    build_horizon_feature_matrix,
)
from .sentiment import (
    score_comments_vader,
    extract_dual_stage_sentiment_features,
)

__all__ = [
    "clean_videos_and_cohort",
    "extract_static_features",
    "extract_temporal_features",
    "build_horizon_feature_matrix",
    "score_comments_vader",
    "extract_dual_stage_sentiment_features",
]
