import os
import sys
import json
import cv2
from datetime import datetime
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv.tracking.zone_mapper import assign_to_zones
from cv.tracking.simple_tracker import SimpleTracker
from cv.tracking.state_tracker import ZoneStateTracker
from cv.utils.draw import draw_rect_zone


def load_zones(path="config/zones.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["zones"]


def main():
    # ---- tuning knobs for testing debounce ----
    min_unknown_frames = 6   # track must exist this many frames before counting/displaying
    min_stable_frames = 5    # zone count must be stable this many frames before "change" triggers

    save_dir = "captured_errors"
    os.makedirs(save_dir, exist_ok=True)

    weights = "runs/inventory/spool_detector_v15/weights/best.pt"
    model = YOLO(weights)

    zones = load_zones("config/zones.json")

    tracker = SimpleTracker(max_age_frames=60, match_dist_px=250.0, max_zone_gap_frames=20, enforce_same_label=False)
    state_tracker = ZoneStateTracker(zones, min_stable_frames=min_stable_frames)

    # track_age: track_id -> {"zone_id": zid, "age": int}
    track_age = {}

    cap = cv2.VideoCapture(2)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    print("Interface Started:")
    print("- Press 'q' to quit")
    print("- Press 'k' to capture the current frame")
    print(f"- min_unknown_frames={min_unknown_frames}, min_stable_frames={min_stable_frames}")

    frame_i = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_i += 1
        raw_frame = frame.copy()

        # draw zones
        for z in zones:
            draw_rect_zone(frame, z)

        results = model.predict(frame, conf=0.4, verbose=False)
        r = results[0]

        dets = []
        if r.boxes is not None:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cf = float(b.conf.item())
                dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "label": "filament_spool",
                    "conf": cf
                })

        # assign dets to zones
        dets = assign_to_zones(dets, zones)

        # tracking
        tracks_out, transfers, enters, exits = tracker.update(dets)

        # update track_age + only keep "eligible" tracks after min_unknown_frames
        seen_ids = set()
        eligible_tracks = []
        for t in tracks_out:
            tid = t["track_id"]
            zid = t.get("zone_id")
            seen_ids.add(tid)

            prev = track_age.get(tid)
            if prev is None or prev["zone_id"] != zid:
                track_age[tid] = {"zone_id": zid, "age": 1}
            else:
                prev["age"] += 1

            if track_age[tid]["age"] >= min_unknown_frames:
                eligible_tracks.append(t)

        # cleanup ages for tracks no longer present
        stale = [tid for tid in track_age.keys() if tid not in seen_ids]
        for tid in stale:
            del track_age[tid]

        # compute zone counts from eligible tracks
        counts = {z["zone_id"]: 0 for z in zones}
        for t in eligible_tracks:
            zid = t.get("zone_id")
            if zid in counts:
                counts[zid] += 1

        # stable changes (debounced)
        changes = state_tracker.update(counts)

        # draw eligible tracks
        for t in eligible_tracks:
            x1, y1, x2, y2 = t["bbox"]
            tid = t["track_id"]
            zid = t.get("zone_id")
            age = track_age.get(tid, {}).get("age", 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(frame, f"#{tid} spool age={age} {zid}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # show counts + changes
        y = 30
        for zid, c in counts.items():
            cv2.putText(frame, f"{zid}: {c}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y += 25

        if changes:
            cv2.putText(frame, f"STABLE CHANGES: {changes}", (20, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("Spool Detector (debounced)", frame)

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