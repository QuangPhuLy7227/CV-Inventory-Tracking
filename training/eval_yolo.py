from ultralytics import YOLO

def main(
    weights="runs/inventory/yolov8s_filament_printer_v1/weights/best.pt",
    data_yaml="datasets/inventory_v1/data.yaml",
    imgsz=640,
    device="cpu"
):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device)
    print(metrics)

if __name__ == "__main__":
    main()