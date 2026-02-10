import cv2
from ultralytics import YOLO

def main(
    weights="runs/inventory/yolov8s_filament_printer_v1/weights/best.pt",
    conf=0.25,
    cam_index=0
):
    model = YOLO(weights)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=conf, verbose=False)
        r = results[0]

        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                cls = int(b.cls.item())
                cf = float(b.conf.item())
                label = r.names.get(cls, str(cls))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)
                cv2.putText(frame, f"{label} {cf:.2f}", (int(x1), int(y1)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Infer Live", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()