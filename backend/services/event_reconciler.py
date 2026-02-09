from __future__ import annotations
from typing import Optional
from backend.models.events import CVZoneChangeEvent, QRScanEvent, PendingConfirmation
from backend.models.common import InventoryObjectType
from backend.services.inventory_state_engine import InventoryStateEngine
from backend.services.confirmation_manager import ConfirmationManager

class EventReconciler:
    """
    Phase 1 reconciliation rules:
    - CV events create "pending" items unless identity is strongly confirmed elsewhere.
    - QR scan events can commit immediately (strong identity).
    - Human confirmation can commit a pending move; if hinted_object_id exists, use it;
      otherwise confirmation must provide the object_id.
    """

    def __init__(self, engine: InventoryStateEngine, confirmations: ConfirmationManager):
        self.engine = engine
        self.confirmations = confirmations

    def ingest_cv(self, ev: CVZoneChangeEvent) -> PendingConfirmation:
        # CV movement is a signal -> create pending confirmation
        return self.confirmations.create_pending(
            object_type=ev.object_type,
            from_zone=ev.from_zone,
            to_zone=ev.to_zone,
            hinted_object_id=ev.hinted_object_id,
        )

    def ingest_qr(self, ev: QRScanEvent) -> None:
        # QR scan is strong identity -> commit what we can
        if ev.scanned_type == InventoryObjectType.filament_spool:
            # If context is printer, you can mount; else update location
            if ev.context_printer_id:
                self.engine.commit_mount(spool_id=ev.scanned_id, printer_id=ev.context_printer_id, zone_id=ev.context_zone)
            else:
                self.engine.commit_location_change(
                    object_type=ev.scanned_type,
                    object_id=ev.scanned_id,
                    to_zone=ev.context_zone,
                )
        elif ev.scanned_type == InventoryObjectType.printer:
            self.engine.commit_location_change(
                object_type=ev.scanned_type,
                object_id=ev.scanned_id,
                to_zone=ev.context_zone,
            )

    def confirm_pending(self, pending_id: str, object_id: Optional[str], resolved_by: str) -> PendingConfirmation:
        pc = self.confirmations.confirm(pending_id, resolved_by=resolved_by)
        # commit after confirmation
        commit_id = pc.hinted_object_id or object_id
        if not commit_id:
            # can't commit without identity
            return pc

        self.engine.commit_location_change(
            object_type=pc.object_type,
            object_id=commit_id,
            to_zone=pc.to_zone,
            from_zone=pc.from_zone,
        )
        return pc