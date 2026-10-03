import yaml
from pathlib import Path
from pydantic import BaseModel
from pydantic_settings import BaseSettings

class TemporalConfig(BaseModel):
    observation_window_hours: int = 6
    prediction_window_hours: int = 48
    snapshot_hours: list[int] = [1, 3, 6]

class Settings(BaseSettings):
    youtube_api_key: str | None = None  # Read from .env
    temporal: TemporalConfig = TemporalConfig()
    
    @classmethod
    def load_from_yaml(cls, path: str = "configs/config.yaml") -> "Settings":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

