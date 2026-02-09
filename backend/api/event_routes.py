from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List

from backend.models.events import CVZoneChangeEvent, QRScanEvent, PendingConfirmation
from backend.api.inventory_routes import ENGINE, STORE  # reuse Phase 1 singletons
from backend.services.confirmation_manager import ConfirmationManager
from backend.services.event_reconciler import EventReconciler

router = APIRouter()

CONFIRMATIONS = ConfirmationManager()
RECONCILER = EventReconciler(engine=ENGINE, confirmations=CONFIRMATIONS)

def _save_inventory_only():
    STORE.save({
        "zones": [z.model_dump() for z in ENGINE.zones.values()],
        "spools": [s.model_dump() for s in ENGINE.spools.values()],
        "printers": [p.model_dump() for p in ENGINE.printers.values()],
    })

class ConfirmRequest(BaseModel):
    resolved_by: str = Field(..., examples=["worker_1", "admin"])
    object_id: Optional[str] = Field(default=None, description="Required if CV had no hinted_object_id")
    note: Optional[str] = None

# ---- ingest events ----
@router.post("/events/cv", response_model=PendingConfirmation)
def ingest_cv(ev: CVZoneChangeEvent):
    pending = RECONCILER.ingest_cv(ev)
    return pending

@router.post("/events/qr")
def ingest_qr(ev: QRScanEvent):
    RECONCILER.ingest_qr(ev)
    _save_inventory_only()
    return {"ok": True}

# ---- confirmations ----
@router.get("/confirmations/pending", response_model=List[PendingConfirmation])
def list_pending():
    return list(CONFIRMATIONS.list_pending().values())

@router.post("/confirmations/{pending_id}/confirm", response_model=PendingConfirmation)
def confirm_pending(pending_id: str, req: ConfirmRequest):
    try:
        pc = RECONCILER.confirm_pending(pending_id, object_id=req.object_id, resolved_by=req.resolved_by)
        _save_inventory_only()
        return pc
    except KeyError:
        raise HTTPException(status_code=404, detail="pending_id not found")

@router.post("/confirmations/{pending_id}/reject", response_model=PendingConfirmation)
def reject_pending(pending_id: str, req: ConfirmRequest):
    try:
        pc = CONFIRMATIONS.reject(pending_id, resolved_by=req.resolved_by, note=req.note)
        return pc
    except KeyError:
        raise HTTPException(status_code=404, detail="pending_id not found")