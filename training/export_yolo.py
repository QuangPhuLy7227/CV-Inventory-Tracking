from ultralytics import YOLO

def main(
    weights="runs/inventory/yolov8s_filament_printer_v1/weights/best.pt",
    fmt="torchscript"
):
    model = YOLO(weights)
    model.export(format=fmt)  # torchscript, onnx, openvino, etc.

if __name__ == "__main__":
    main()