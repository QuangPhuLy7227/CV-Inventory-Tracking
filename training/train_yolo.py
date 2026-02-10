from ultralytics import YOLO

def main(
    data_yaml="datasets/inventory_v1/data.yaml",
    model="yolov8s.pt",
    epochs=60,
    imgsz=640,
    batch=8,
    device="cpu",
    project="runs/inventory",
    name="yolov8s_filament_printer_v1"
):
    yolo = YOLO(model)
    yolo.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name
    )

if __name__ == "__main__":
    main()