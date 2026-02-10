# Inventory Tracking - Phase 1

Phase 1 builds the inventory truth model + backend APIs.
- CV events are treated as signals (never directly commit inventory)
- Human confirmation (QR scan or manual confirm) is required to commit uncertain moves

## Run
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000
```

# Inventory Tracking - Phase 2

Phase 2 builds Computer Visin model.
- CV detect objects and objects count within the zone
- Publish an event when it is APPEAR/DISAPPEAR/TRANSFER

## Run
```bash
python -m cv.main
```

# Phase 3 - No None gap
- Phase 3 adding the feature for no none gap when switching between zones
- And add functionalities to detect object entering or exiting events

# Phase 4 - Training YOLO
- Start with the generic YOLOv8 model
- Collect images through 3 ways: self collection with video capturing, manual import, or downloading from an url
- Labeling using Roboflow Annotate
- Prep data using scikit learn
- Infer live for testing

# Phase 5 - Adding QR Reader
- Using QR-pillow to generate QR from JSON
- Decode QR using OpenCV
```bash
python -m training.test_qr_live
```