import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any

class EventPublisher:
    def __init__(self, base_url: str, path: str, timeout_seconds: int = 2):
        self.url = base_url.rstrip("/") + path
        self.timeout = timeout_seconds

    def publish_zone_change(
        self,
        object_type: str,
        from_zone: Optional[str],
        to_zone: Optional[str],
        hinted_object_id: Optional[str] = None,
        confidence: float = 0.6,
        meta: Optional[Dict[str, Any]] = None,
    ):
        payload = {
            "event_type": "cv_zone_change",
            "object_type": object_type,
            "from_zone": from_zone,
            "to_zone": to_zone,
            "hinted_object_id": hinted_object_id,
            "confidence": confidence,
            "meta": meta or {},
            "timestamp": datetime.utcnow().isoformat(),
        }
        r = requests.post(self.url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()