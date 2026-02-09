import unittest
from backend.services.inventory_state_engine import InventoryStateEngine
from backend.services.confirmation_manager import ConfirmationManager
from backend.services.event_reconciler import EventReconciler
from backend.models.events import CVZoneChangeEvent, QRScanEvent
from backend.models.common import InventoryObjectType
from backend.models.inventory import Zone, FilamentSpool

class TestPhase1Rules(unittest.TestCase):
    def setUp(self):
        self.engine = InventoryStateEngine()
        self.confirm = ConfirmationManager()
        self.recon = EventReconciler(self.engine, self.confirm)

        self.engine.upsert_zone(Zone(zone_id="Rack_A_Slot_1"))
        self.engine.upsert_zone(Zone(zone_id="Printer_P3_Mount"))
        self.engine.upsert_spool(FilamentSpool(spool_id="SPOOL-1", zone_id="Rack_A_Slot_1"))

    def test_cv_event_creates_pending_not_commit(self):
        ev = CVZoneChangeEvent(
            object_type=InventoryObjectType.filament_spool,
            from_zone="Rack_A_Slot_1",
            to_zone="Printer_P3_Mount",
            hinted_object_id=None
        )
        pending = self.recon.ingest_cv(ev)
        # spool location should NOT change yet
        self.assertEqual(self.engine.spools["SPOOL-1"].zone_id, "Rack_A_Slot_1")
        self.assertEqual(pending.status, "pending")

    def test_confirm_commits_location(self):
        ev = CVZoneChangeEvent(
            object_type=InventoryObjectType.filament_spool,
            from_zone="Rack_A_Slot_1",
            to_zone="Printer_P3_Mount",
            hinted_object_id="SPOOL-1"
        )
        pending = self.recon.ingest_cv(ev)
        self.recon.confirm_pending(pending.pending_id, object_id=None, resolved_by="worker")
        self.assertEqual(self.engine.spools["SPOOL-1"].zone_id, "Printer_P3_Mount")

    def test_qr_scan_commits_location(self):
        qr = QRScanEvent(
            scanned_id="SPOOL-1",
            scanned_type=InventoryObjectType.filament_spool,
            context_zone="Printer_P3_Mount",
        )
        self.recon.ingest_qr(qr)
        self.assertEqual(self.engine.spools["SPOOL-1"].zone_id, "Printer_P3_Mount")

if __name__ == "__main__":
    unittest.main()