import cv2
from cv.qr.qr_reader import QRReader

PERSIST_FRAMES = 20  # keep drawing a QR box for this many frames after last detection

def main(cam_index=1):
    qr = QRReader()
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera.")

    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 30)

    print("Show QR codes to camera. Press 'q' to quit.")

    # cache: raw_string -> {"bbox": [...], "label": str, "last_frame": int}
    qr_cache = {}
    frame_i = 0
    last_printed = set()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_i += 1

        qr_list = qr.decode_multi_bgr(frame)

        # update cache with fresh detections
        for item in qr_list:
            raw = item.get("raw")
            if not raw:
                continue
            payload = item.get("payload")
            bbox = item.get("bbox")
            label = "QR"
            if isinstance(payload, dict) and payload.get("id"):
                label = f"QR:{payload['id']}"

            if raw not in last_printed:
                print("\n[QR] raw:", raw)
                if payload:
                    print("[QR] parsed:", payload)
                last_printed.add(raw)

            if bbox:
                qr_cache[raw] = {"bbox": bbox, "label": label, "last_frame": frame_i}

        # draw all cached QRs that are still within PERSIST_FRAMES
        stale = []
        for raw, entry in qr_cache.items():
            if (frame_i - entry["last_frame"]) > PERSIST_FRAMES:
                stale.append(raw)
                continue
            x1, y1, x2, y2 = entry["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, entry["label"], (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for raw in stale:
            del qr_cache[raw]
            last_printed.discard(raw)

        cv2.imshow("QR Test (Multi)", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
