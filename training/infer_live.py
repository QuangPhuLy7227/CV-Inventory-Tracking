import cv2
from ultralytics import YOLO

def main():
    weights = "runs/inventory/spool_detector_v1/weights/best.pt"
    model = YOLO(weights)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=0.25, verbose=False)
        r = results[0]
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cf = float(b.conf.item())
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), 2)
                cv2.putText(frame, f"filament_spool {cf:.2f}", (x1, max(0,y1-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Spool Detector", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()