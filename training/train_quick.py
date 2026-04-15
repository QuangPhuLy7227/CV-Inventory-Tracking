from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # COCO pretrained starting weights
    model.train(
        data="datasets/inventory_v1/data.yaml",
        epochs=20,          # quick
        imgsz=640,
        batch=8,
        device="cpu",
        project="runs/inventory",
        name="spool_quick_v1",  # unique name => no overwrite
        pretrained=True
    )

if __name__ == "__main__":
    main()