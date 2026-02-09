from __future__ import annotations
from typing import Dict, Optional
from datetime import datetime, timezone
from backend.models.inventory import FilamentSpool, Printer, Zone
from backend.models.common import InventoryObjectType

class InventoryStateEngine:
    """
    Single source of truth rules (Phase 1):
    - CV never commits identity changes directly.
    - Confirmed QR scans can commit.
    - Confirmations can commit a move from zone A -> zone B.
    """

    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        self.spools: Dict[str, FilamentSpool] = {}
        self.printers: Dict[str, Printer] = {}

    # ---------- CRUD ----------
    def upsert_zone(self, z: Zone) -> Zone:
        self.zones[z.zone_id] = z
        return z

    def delete_zone(self, zone_id: str) -> None:
        self.zones.pop(zone_id, None)

    def upsert_spool(self, s: FilamentSpool) -> FilamentSpool:
        s.updated_at = datetime.now(timezone.utc)
        self.spools[s.spool_id] = s
        return s

    def delete_spool(self, spool_id: str) -> None:
        self.spools.pop(spool_id, None)

    def upsert_printer(self, p: Printer) -> Printer:
        p.updated_at = datetime.now(timezone.utc)
        self.printers[p.printer_id] = p
        return p

    def delete_printer(self, printer_id: str) -> None:
        self.printers.pop(printer_id, None)

    # ---------- Commits ----------
    def commit_location_change(
        self,
        object_type: InventoryObjectType,
        object_id: str,
        to_zone: Optional[str],
        from_zone: Optional[str] = None,
    ) -> None:
        """
        Apply a confirmed location change. For Phase 1 we only care count+location,
        so this updates the object's zone_id.
        """
        if object_type == InventoryObjectType.filament_spool:
            if object_id not in self.spools:
                # auto-create minimal object if it doesn't exist yet (optional policy)
                self.spools[object_id] = FilamentSpool(spool_id=object_id, zone_id=None)
            s = self.spools[object_id]
            # Optional check: if from_zone provided and differs, we still allow commit
            s.zone_id = to_zone
            s.updated_at = datetime.utcnow()

        elif object_type == InventoryObjectType.printer:
            if object_id not in self.printers:
                self.printers[object_id] = Printer(printer_id=object_id, zone_id=None)
            p = self.printers[object_id]
            p.zone_id = to_zone
            p.updated_at = datetime.utcnow()

    def commit_mount(
        self,
        spool_id: str,
        printer_id: str,
        zone_id: Optional[str] = None,
    ) -> None:
        """
        Optional for later: link spool to printer.
        Still Phase 1-safe (doesn't require CV).
        """
        if spool_id not in self.spools:
            self.spools[spool_id] = FilamentSpool(spool_id=spool_id, zone_id=zone_id)
        if printer_id not in self.printers:
            self.printers[printer_id] = Printer(printer_id=printer_id, zone_id=None)

        s = self.spools[spool_id]
        p = self.printers[printer_id]
        s.mounted_printer_id = printer_id
        p.mounted_spool_id = spool_id

        # if provided, update zone too
        if zone_id:
            s.zone_id = zone_id

        s.updated_at = datetime.now(timezone.utc)
        p.updated_at = datetime.now(timezone.utc)