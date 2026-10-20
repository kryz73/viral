import os
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectConfig(BaseModel):
    name: str = "viral"
    version: str = "0.1.0"
    random_seed: int = 42


class PathsConfig(BaseModel):
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed"
    features_data_dir: str = "data/features"
    models_dir: str = "models"
    reports_dir: str = "reports"


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./data/viral.db"
    echo_sql: bool = False


class TemporalConfig(BaseModel):
    observation_window_hours: int = 6
    prediction_window_hours: int = 48
    snapshot_hours: List[int] = Field(default_factory=lambda: [1, 3, 6])


class VaderConfig(BaseModel):
    pos_threshold: float = 0.05
    neg_threshold: float = -0.05


class SpamFilterConfig(BaseModel):
    min_length: int = 3
    max_repeated_chars: int = 4


class NLPConfig(BaseModel):
    vader: VaderConfig = Field(default_factory=VaderConfig)
    transformer_model: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    batch_size: int = 64
    max_comment_length: int = 256
    spam_filter: SpamFilterConfig = Field(default_factory=SpamFilterConfig)


class ModelsConfig(BaseModel):
    primary_metric: str = "pr_auc"
    time_split_ratios: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.65, "validation": 0.15, "test": 0.20}
    )
    ablation_experiments: Dict[str, Any] = Field(default_factory=dict)


class MLflowConfig(BaseModel):
    tracking_uri: str = "sqlite:///./mlruns.db"
    experiment_name: str = "youtube_trending_ablation"


class CrawlerConfig(BaseModel):
    enabled: bool = True
    channel_poll_interval_hours: int = 2
    snapshot_poll_interval_minutes: int = 10
    comment_harvest_interval_minutes: int = 15
    trending_check_interval_hours: int = 2
    max_comments_per_video: int = 250
    regions: List[str] = Field(default_factory=lambda: ["US", "GB", "CA", "IN"])
    target_categories: List[int] = Field(default_factory=lambda: [10, 20, 24, 25, 28])
    seed_channels_path: str = "configs/seed_channels.json"
    s3_backup_interval_hours: int = 6


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    youtube_api_key: Optional[str] = None
    youtube_api_keys: Optional[str] = None
    database_url: Optional[str] = None
    aws_s3_bucket: Optional[str] = None

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    temporal_windows: TemporalConfig = Field(default_factory=TemporalConfig)
    nlp: NLPConfig = Field(default_factory=NLPConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    crawler: CrawlerConfig = Field(default_factory=CrawlerConfig)

    def get_api_keys(self) -> List[str]:
        """Return all available YouTube API keys (supporting single or comma-separated keys)."""
        keys: List[str] = []
        if self.youtube_api_keys:
            keys.extend([k.strip() for k in self.youtube_api_keys.split(",") if k.strip()])
        if self.youtube_api_key and self.youtube_api_key not in keys:
            keys.append(self.youtube_api_key.strip())
        return keys

    @classmethod
    def load_from_yaml(cls, path: str = "configs/config.yaml") -> "Settings":
        yaml_path = Path(path)
        if yaml_path.exists():
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)
        return cls()


# Global settings singleton
settings = Settings.load_from_yaml()
