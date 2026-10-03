"""
Database ORM Models for YouTube Trending Prediction System.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    String,
    Text,
    Integer,
    BigInteger,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.storage.db import Base


# =====================================================================
# 1. Video Metadata
# =====================================================================
class Video(Base):
    __tablename__ = "videos"

    video_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    channel_id: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    category_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    observations: Mapped[List["VideoObservation"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    comments: Mapped[List["Comment"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    features: Mapped[List["VideoFeatures"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )
    predictions: Mapped[List["ModelPrediction"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Video id={self.video_id} title='{self.title[:30]}...'>"


# =====================================================================
# 2. Periodic Video Observations (Snapshots)
# =====================================================================
class VideoObservation(Base):
    __tablename__ = "video_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("videos.video_id", ondelete="CASCADE"), index=True, nullable=False
    )
    observed_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    views: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    likes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    comments: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    is_trending: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationship
    video: Mapped["Video"] = relationship(back_populates="observations")

    def __repr__(self) -> str:
        return f"<Observation video={self.video_id} views={self.views} trending={self.is_trending}>"


# =====================================================================
# 3. Audience Comments
# =====================================================================
class Comment(Base):
    __tablename__ = "comments"

    comment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("videos.video_id", ondelete="CASCADE"), index=True, nullable=False
    )
    published_at: Mapped[datetime] = mapped_column(DateTime, index=True, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    likes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    replies: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Relationships
    video: Mapped["Video"] = relationship(back_populates="comments")
    analysis: Mapped[Optional["CommentAnalysis"]] = relationship(
        back_populates="comment", uselist=False, cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Comment id={self.comment_id} video={self.video_id}>"


# =====================================================================
# 4. Comment NLP Analysis Output
# =====================================================================
class CommentAnalysis(Base):
    __tablename__ = "comment_analyses"

    comment_id: Mapped[str] = mapped_column(
        ForeignKey("comments.comment_id", ondelete="CASCADE"), primary_key=True
    )
    sentiment: Mapped[str] = mapped_column(String(16), nullable=False)  # positive, neutral, negative
    vader_compound: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    vader_pos: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    vader_neg: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    vader_neu: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    distilbert_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_spam: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    spam_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship
    comment: Mapped["Comment"] = relationship(back_populates="analysis")


# =====================================================================
# 5. Engineered Video Features (Calculated at t=Cutoff)
# =====================================================================
class VideoFeatures(Base):
    __tablename__ = "video_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("videos.video_id", ondelete="CASCADE"), index=True, nullable=False
    )
    observation_cutoff_hours: Mapped[float] = mapped_column(Float, default=6.0, nullable=False)

    # Static Metadata Features
    title_length: Mapped[int] = mapped_column(Integer, default=0)
    description_length: Mapped[int] = mapped_column(Integer, default=0)
    tag_count: Mapped[int] = mapped_column(Integer, default=0)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)
    category_id: Mapped[int] = mapped_column(Integer, default=0)
    published_hour: Mapped[int] = mapped_column(Integer, default=0)
    published_day: Mapped[int] = mapped_column(Integer, default=0)

    # Early Engagement & Velocity Features (t <= Cutoff)
    views_cutoff: Mapped[int] = mapped_column(BigInteger, default=0)
    likes_cutoff: Mapped[int] = mapped_column(Integer, default=0)
    comments_cutoff: Mapped[int] = mapped_column(Integer, default=0)
    view_velocity: Mapped[float] = mapped_column(Float, default=0.0)
    like_velocity: Mapped[float] = mapped_column(Float, default=0.0)
    comment_velocity: Mapped[float] = mapped_column(Float, default=0.0)
    like_rate: Mapped[float] = mapped_column(Float, default=0.0)
    comment_rate: Mapped[float] = mapped_column(Float, default=0.0)

    # Early Acceleration (Growth rate changes between 1h -> 3h -> 6h)
    view_acceleration: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Audience Sentiment Aggregations
    mean_sentiment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    positive_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    negative_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    neutral_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_variance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_change: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spam_comment_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Future Ground-Truth Target Label (1 if trending in 6h-54h, 0 otherwise)
    is_trending_future: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationship
    video: Mapped["Video"] = relationship(back_populates="features")


# =====================================================================
# 6. Model Prediction Logs
# =====================================================================
class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    video_id: Mapped[str] = mapped_column(
        ForeignKey("videos.video_id", ondelete="CASCADE"), index=True, nullable=False
    )
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    prediction_time: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    trending_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Store top SHAP feature drivers as JSON: {"view_velocity": +0.32, "like_rate": -0.12}
    top_features: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Relationship
    video: Mapped["Video"] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return f"<Prediction video={self.video_id} prob={self.trending_probability:.2f}>"