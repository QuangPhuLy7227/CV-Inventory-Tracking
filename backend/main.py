from fastapi import FastAPI
from backend.api.health_routes import router as health_router
from backend.api.inventory_routes import router as inventory_router
from backend.api.event_routes import router as event_router
from backend.core.logging import setup_logging

setup_logging()

app = FastAPI(
    title="Inventory Tracking Backend (Phase 1)",
    version="1.0.0",
)

app.include_router(health_router)
app.include_router(inventory_router, prefix="/api")
app.include_router(event_router, prefix="/api")