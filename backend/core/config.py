from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="INV_", env_file=".env", extra="ignore")

    storage_path: str = Field(default="backend_state.json", description="JSON persistence file")
    pending_timeout_seconds: int = Field(default=20, description="How long to wait for QR scan after CV movement")

settings = Settings()