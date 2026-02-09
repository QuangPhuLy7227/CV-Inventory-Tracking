from fastapi import APIRouter, HTTPException
from backend.models.inventory import Zone, FilamentSpool, Printer
from backend.services.inventory_state_engine import InventoryStateEngine
from backend.services.storage import JsonStateStore

router = APIRouter()

# Simple singleton instances for Phase 1 demo
ENGINE = InventoryStateEngine()
STORE = JsonStateStore()

def _load_once():
    state = STORE.load()
    if not state:
        return
    for z in state.get("zones", []):
        ENGINE.upsert_zone(Zone(**z))
    for s in state.get("spools", []):
        ENGINE.upsert_spool(FilamentSpool(**s))
    for p in state.get("printers", []):
        ENGINE.upsert_printer(Printer(**p))

def _save():
    STORE.save({
        "zones": [z.model_dump() for z in ENGINE.zones.values()],
        "spools": [s.model_dump() for s in ENGINE.spools.values()],
        "printers": [p.model_dump() for p in ENGINE.printers.values()],
    })

_load_once()

# ---------- Zones ----------
@router.get("/zones")
def list_zones():
    return list(ENGINE.zones.values())

@router.post("/zones")
def upsert_zone(zone: Zone):
    z = ENGINE.upsert_zone(zone)
    _save()
    return z

@router.delete("/zones/{zone_id}")
def delete_zone(zone_id: str):
    ENGINE.delete_zone(zone_id)
    _save()
    return {"deleted": zone_id}

# ---------- Spools ----------
@router.get("/spools")
def list_spools():
    return list(ENGINE.spools.values())

@router.post("/spools")
def upsert_spool(spool: FilamentSpool):
    if spool.zone_id and spool.zone_id not in ENGINE.zones:
        raise HTTPException(status_code=400, detail="zone_id does not exist")
    s = ENGINE.upsert_spool(spool)
    _save()
    return s

@router.delete("/spools/{spool_id}")
def delete_spool(spool_id: str):
    ENGINE.delete_spool(spool_id)
    _save()
    return {"deleted": spool_id}

# ---------- Printers ----------
@router.get("/printers")
def list_printers():
    return list(ENGINE.printers.values())

@router.post("/printers")
def upsert_printer(printer: Printer):
    if printer.zone_id and printer.zone_id not in ENGINE.zones:
        raise HTTPException(status_code=400, detail="zone_id does not exist")
    p = ENGINE.upsert_printer(printer)
    _save()
    return p

@router.delete("/printers/{printer_id}")
def delete_printer(printer_id: str):
    ENGINE.delete_printer(printer_id)
    _save()
    return {"deleted": printer_id}