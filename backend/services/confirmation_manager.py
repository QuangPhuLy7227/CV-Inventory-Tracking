from __future__ import annotations
from datetime import datetime, timedelta
from typing import Dict, Optional
import uuid

from backend.core.config import settings
from backend.models.events import PendingConfirmation
from backend.models.common import InventoryObjectType

class ConfirmationManager:
    def __init__(self):
        self.pending: Dict[str, PendingConfirmation] = {}

    def create_pending(
        self,
        object_type: InventoryObjectType,
        from_zone: Optional[str],
        to_zone: Optional[str],
        hinted_object_id: Optional[str],
    ) -> PendingConfirmation:
        pending_id = str(uuid.uuid4())
        now = datetime.utcnow()
        expires_at = now + timedelta(seconds=settings.pending_timeout_seconds)
        pc = PendingConfirmation(
            pending_id=pending_id,
            object_type=object_type,
            from_zone=from_zone,
            to_zone=to_zone,
            hinted_object_id=hinted_object_id,
            created_at=now,
            expires_at=expires_at,
        )
        self.pending[pending_id] = pc
        return pc

    def list_pending(self) -> Dict[str, PendingConfirmation]:
        self.expire_old()
        return self.pending

    def expire_old(self) -> None:
        now = datetime.utcnow()
        for pid, pc in list(self.pending.items()):
            if pc.status == "pending" and pc.expires_at <= now:
                pc.status = "expired"
                pc.resolved_at = now
                pc.resolution_note = "Auto-expired"
                self.pending[pid] = pc

    def confirm(self, pending_id: str, resolved_by: str, note: str | None = None) -> PendingConfirmation:
        self.expire_old()
        pc = self.pending.get(pending_id)
        if not pc:
            raise KeyError("pending_id not found")
        if pc.status != "pending":
            return pc

        pc.status = "confirmed"
        pc.resolved_by = resolved_by
        pc.resolved_at = datetime.utcnow()
        pc.resolution_note = note
        self.pending[pending_id] = pc
        return pc

    def reject(self, pending_id: str, resolved_by: str, note: str | None = None) -> PendingConfirmation:
        self.expire_old()
        pc = self.pending.get(pending_id)
        if not pc:
            raise KeyError("pending_id not found")
        if pc.status != "pending":
            return pc

        pc.status = "rejected"
        pc.resolved_by = resolved_by
        pc.resolved_at = datetime.utcnow()
        pc.resolution_note = note
        self.pending[pending_id] = pc
        return pc