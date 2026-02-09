from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime, timezone
from backend.models.common import ZoneType

class Zone(BaseModel):
    zone_id: str = Field(..., examples=["Rack_A_Slot_1", "Printer_P3_Mount"])
    zone_type: ZoneType = ZoneType.other
    description: Optional[str] = None

    # Geometry will be added in Phase 2 (ROI polygons). Keep placeholder now.
    geometry: Optional[Dict] = Field(default=None, description="Zone geometry placeholder for Phase 2")

class FilamentSpool(BaseModel):
    spool_id: str = Field(..., examples=["SPOOL-PLA-BLK-014"])
    material: Optional[str] = Field(default=None, examples=["PLA", "PETG"])
    color: Optional[str] = Field(default=None, examples=["Black"])
    brand: Optional[str] = None

    # For Phase 1: focus on count + location
    zone_id: Optional[str] = Field(default=None, description="Current location zone_id")
    mounted_printer_id: Optional[str] = None

    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))

class Printer(BaseModel):
    printer_id: str = Field(..., examples=["P3"])
    model: Optional[str] = Field(default=None, examples=["Prusa MK4", "Bambu X1C"])

    zone_id: Optional[str] = Field(default=None, description="Where the printer is located (if tracked by zone)")
    mounted_spool_id: Optional[str] = None

    updated_at: datetime = Field(default_factory=datetime.now(timezone.utc))