from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Literal
from datetime import datetime
from backend.models.common import InventoryObjectType

class CVZoneChangeEvent(BaseModel):
    """
    Produced by CV pipeline later:
    - "something moved" from zone A to zone B (identity optional)
    """
    event_type: Literal["cv_zone_change"] = "cv_zone_change"
    object_type: InventoryObjectType
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    # Optional hint if QR was visible in video (best-effort)
    hinted_object_id: Optional[str] = Field(default=None, description="Soft identity hint from QR-in-video")
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    meta: Dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class QRScanEvent(BaseModel):
    """
    Produced by the worker scanner app (human-in-the-loop):
    - Strong identity confirmation
    """
    event_type: Literal["qr_scan"] = "qr_scan"
    scanned_id: str = Field(..., description="e.g., spool_id or printer_id")
    scanned_type: InventoryObjectType
    # Optional: what they scanned it WITH (printer, rack slot, etc.)
    context_zone: Optional[str] = None
    context_printer_id: Optional[str] = None
    meta: Dict = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PendingConfirmation(BaseModel):
    pending_id: str
    object_type: InventoryObjectType
    from_zone: Optional[str] = None
    to_zone: Optional[str] = None
    hinted_object_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime

    status: Literal["pending", "confirmed", "rejected", "expired"] = "pending"
    resolved_by: Optional[str] = None  # e.g., operator or scanned id
    resolved_at: Optional[datetime] = None
    resolution_note: Optional[str] = None