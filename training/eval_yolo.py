from ultralytics import YOLO

def main(
    weights="runs/inventory/spool_detector_v15/weights/best.pt",
    data_yaml="datasets/inventory_v1/data.local.yaml",
    imgsz=640,
    device="cpu",
    split="test",   # "val" for quick check, "test" for final reported score
):
    model = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=imgsz, device=device, split=split)
    print(metrics)

if __name__ == "__main__":
    main()