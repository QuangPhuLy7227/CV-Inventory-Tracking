import json
import yaml

from cv.detectors.yolo_detector import YOLODetector
from cv.tracking.zone_mapper import assign_to_zones, count_by_zone
from cv.tracking.state_tracker import ZoneStateTracker, infer_transfers
from cv.utils.draw import draw_rect_zone, draw_bbox
from cv.tracking.simple_tracker import SimpleTracker


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_zones(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["zones"]


class CVPipeline:
    def __init__(self, cv_config_path="config/cv.yaml", zones_path="config/zones.json"):
        cfg = load_yaml(cv_config_path)
        self.cfg = cfg
        self.zones = load_zones(zones_path)

        self.detector = YOLODetector(
            model_path=cfg["yolo"]["model"],
            conf=float(cfg["yolo"]["conf"]),
            iou=float(cfg["yolo"]["iou"]),
            device=str(cfg["yolo"]["device"]),
        )

        self.class_filter = set(cfg.get("detect_classes") or [])
        self.process_every_n = int(cfg["logic"]["process_every_n_frames"])
        self.publish_events = bool(cfg["logic"]["publish_events"])
        self.object_type = "generic_object"  # Phase 2 testing. Later: filament_spool / printer.

        self.tracker = SimpleTracker(max_age_frames=30, match_dist_px=140.0)

        self.state_tracker = ZoneStateTracker(
            self.zones,
            min_stable_frames=int(cfg["logic"]["min_stable_frames"]),
        )

        # Publisher is optional, and imported only if needed
        self.publisher = None
        if self.publish_events:
            from cv.events.event_publisher import EventPublisher  # lazy import
            self.publisher = EventPublisher(
                base_url=cfg["backend"]["base_url"],
                path=cfg["backend"]["cv_event_path"],
                timeout_seconds=int(cfg["backend"]["timeout_seconds"]),
            )

        self.frame_i = 0

    def _filter_dets(self, dets):
        if not self.class_filter:
            return dets
        return [d for d in dets if d["label"] in self.class_filter]

    def step(self, frame_bgr):
        """
        Process one frame. Returns:
          annotated_frame, debug_info
        """
        self.frame_i += 1
        annotated = frame_bgr.copy()

        # Always draw zones
        for z in self.zones:
            draw_rect_zone(annotated, z)

        debug = {"published": [], "counts": None, "changes": [], "transfers": [], "residual": []}

        # Only run detection every N frames to reduce CPU load
        if (self.frame_i % self.process_every_n) != 0:
            return annotated, debug
        
        # 1) Detect
        dets = self.detector.detect(frame_bgr)
        dets = [d for d in dets if d["label"] != "person"]

        # Debug: show what YOLO sees
        # if dets:
        #     print("[YOLO] detections:", [(d["label"], round(d["conf"], 2)) for d in dets])
        # else:
        #     print("[YOLO] detections: []")

        # 2) Filter + zone-assign
        dets = self._filter_dets(dets)
        dets = [d for d in dets if d["conf"] >= 0.35]
        dets = assign_to_zones(dets, self.zones)

        # 3) Tracking-based transfers (best for MOVE events)
        tracks_out, transfers = self.tracker.update(dets)
        debug["transfers"] = transfers

        # 4) Draw tracked boxes w/ IDs
        # draw detections
        # for d in dets:
        #     draw_bbox(annotated, d["bbox"], d["label"], d["conf"])
        for t in tracks_out:
            draw_bbox(annotated, t["bbox"], f"#{t['track_id']} {t['label']}", t["conf"])

        # 5) Zone counts (use tracked objects for stability)
        counts = {z["zone_id"]: 0 for z in self.zones}
        for t in tracks_out:
            zid = t.get("zone_id")
            if zid in counts:
                counts[zid] += 1
        debug["counts"] = counts

        # print("[DEBUG] active tracks:", [(t["track_id"], t["label"], t.get("zone_id")) for t in tracks_out])

        # 6) Debounced APPEAR/DISAPPEAR using ZoneStateTracker
        #    This gives you residual events even when tracking is noisy.
        changes = self.state_tracker.update(counts)
        debug["changes"] = changes

        residual = []
        for c in changes:
            if c["new"] > c["old"]:
                residual.append({
                    "mode": "appearance",
                    "from_zone": None,
                    "to_zone": c["zone_id"],
                    "old": c["old"],
                    "new": c["new"],
                })
            else:
                residual.append({
                    "mode": "disappearance",
                    "from_zone": c["zone_id"],
                    "to_zone": None,
                    "old": c["old"],
                    "new": c["new"],
                })
        debug["residual"] = residual

        # changes = self.state_tracker.update(counts)
        # debug["changes"] = changes

        # if not changes:
        #     return annotated, debug

        # transfers, residual = infer_transfers(changes)
        # debug["transfers"] = transfers

        # 7) Publish or print events
        if self.publish_events and self.publisher is not None:
            # publish inferred transfers
            for t in transfers:
                try:
                    resp = self.publisher.publish_zone_change(
                        object_type=self.object_type,
                        from_zone=t["from_zone"],
                        to_zone=t["to_zone"],
                        hinted_object_id=None,
                        confidence=0.6,
                        meta={"source": "phase2", "mode": "transfer"},
                    )
                    debug["published"].append(resp)
                except Exception as e:
                    debug["published"].append({"error": str(e), "event": t})

            # publish residual appearance/disappearance as partial changes
            for c in residual:
                if c["new"] > c["old"]:
                    from_zone, to_zone = None, c["zone_id"]
                    mode = "appearance"
                else:
                    from_zone, to_zone = c["zone_id"], None
                    mode = "disappearance"

                try:
                    resp = self.publisher.publish_zone_change(
                        object_type=self.object_type,
                        from_zone=from_zone,
                        to_zone=to_zone,
                        hinted_object_id=None,
                        confidence=0.5,
                        meta={"source": "phase2", "mode": mode},
                    )
                    debug["published"].append(resp)
                except Exception as e:
                    debug["published"].append({"error": str(e), "event": c})
        else:
            # Standalone mode: print the events
            for t in transfers:
                print(f"[CV] TRANSFER {t['from_zone']} -> {t['to_zone']}")
            for r in residual:
                if r["mode"] == "appearance":
                    print(f"[CV] APPEAR {r['to_zone']} ({r['old']} -> {r['new']})")
                else:
                    print(f"[CV] DISAPPEAR {r['from_zone']} ({r['old']} -> {r['new']})")

        return annotated, debug