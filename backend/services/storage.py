from __future__ import annotations
import json
import os
from typing import Dict, Any
from backend.core.config import settings

class JsonStateStore:
    """
    Simple JSON persistence so Phase 1 can demo without a DB.
    Later you can replace this with Postgres/Redis.
    """
    def __init__(self, path: str | None = None):
        self.path = path or settings.storage_path

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, state: Dict[str, Any]) -> None:
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)