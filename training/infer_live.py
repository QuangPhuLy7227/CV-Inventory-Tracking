import cv2
from ultralytics import YOLO
from datetime import datetime
import os

def main():
    save_dir = "captured_errors"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    weights = "runs/inventory/spool_detector_v12/weights/best.pt"
    model = YOLO(weights)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")
    
    print("Interface Started:")
    print("- Press 'q' to quit")
    print("- Press 'k' to capture the current frame")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Create a copy of the raw frame BEFORE drawing boxes on it
        # This is useful if you want to use the image for re-training later
        raw_frame = frame.copy()

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

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("k"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{save_dir}/frame_{timestamp}.jpg"
            
            cv2.imwrite(filename, raw_frame)
            print(f"[INFO] Saved frame to: {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()