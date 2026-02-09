# Inventory Tracking - Phase 1

Phase 1 builds the inventory truth model + backend APIs.
- CV events are treated as signals (never directly commit inventory)
- Human confirmation (QR scan or manual confirm) is required to commit uncertain moves

## Run
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000