import cv2
from cv.qr.qr_reader import QRReader

def main(cam_index=0):
    qr = QRReader()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    print("Show QR codes to camera. Press 'q' to quit.")
    last_seen = set()  # to avoid spamming prints

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        qr_list = qr.decode_multi_bgr(frame)  # ✅ multiple QR decode

        # draw + print
        current_seen = set()
        for item in qr_list:
            raw = item.get("raw")
            payload = item.get("payload")
            bbox = item.get("bbox")

            if raw:
                current_seen.add(raw)

                # print only when newly seen
                if raw not in last_seen:
                    print("\n[QR] raw:", raw)
                    if payload:
                        print("[QR] parsed:", payload)

            # draw bbox if available
            if bbox:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # label: show id if payload dict has it
                label = "QR"
                if isinstance(payload, dict) and payload.get("id"):
                    label = f"QR:{payload['id']}"
                cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        last_seen = current_seen

        cv2.imshow("QR Test (Multi)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()