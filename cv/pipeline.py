import json
import yaml

from cv.detectors.yolo_detector import YOLODetector
from cv.tracking.zone_mapper import assign_to_zones, count_by_zone
from cv.tracking.state_tracker import ZoneStateTracker, infer_transfers
from cv.utils.draw import draw_rect_zone, draw_bbox
from cv.tracking.simple_tracker import SimpleTracker
from cv.qr.qr_reader import QRReader


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

        self.tracker = SimpleTracker(
            max_age_frames=60,
            match_dist_px=250.0,
            max_zone_gap_frames=20,
            enforce_same_label=False
        )
        self.state_tracker = ZoneStateTracker(
            self.zones,
            min_stable_frames=int(cfg["logic"]["min_stable_frames"]),
        )

        # QR code reader
        self.qr_cfg = cfg.get("qr", {}) or {}
        self.qr_enabled = bool(self.qr_cfg.get("enabled", False))
        self.qr_every_n = int(self.qr_cfg.get("decode_every_n_frames", 2))
        self.qr_pad = int(self.qr_cfg.get("roi_pad_px", 14))
        self.qr_draw = bool(self.qr_cfg.get("draw_overlay", True))
        self.qr_reader = QRReader() if self.qr_enabled else None

        # cache last known QR per track so you don't need to decode every frame
        self.track_qr_cache = {}  # track_id -> {"raw": str, "payload": dict}

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

        debug = {"published": [], "counts": None, "changes": [], "transfers": [], "enters": [], "exits": [], "residual": []}

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
        tracks_out, transfers, enters, exits = self.tracker.update(dets)
        print("[DEBUG tracks]", [(t["track_id"], t["label"], t.get("prev_zone_id"), t.get("zone_id")) for t in tracks_out])
        print("[DEBUG events]", "T=", len(transfers), "E=", len(enters), "X=", len(exits))

        # --- QR decode step (ROI-based, fast) ---
        if self.qr_enabled and self.qr_reader is not None and (self.frame_i % self.qr_every_n == 0):
            for t in tracks_out:
                tid = t["track_id"]
                raw, payload = self.qr_reader.decode_roi(frame_bgr, t["bbox"], pad=self.qr_pad)
                if raw:
                    self.track_qr_cache[tid] = {"raw": raw, "payload": payload}

        debug["transfers"] = transfers
        debug["enters"] = enters
        debug["exits"] = exits

        # 4) Draw tracked boxes w/ IDs
        # draw detections
        # for d in dets:
        #     draw_bbox(annotated, d["bbox"], d["label"], d["conf"])
        for t in tracks_out:
            tid = t["track_id"]
            label = t["label"]
            cache = self.track_qr_cache.get(tid)

            if self.qr_draw and cache:
                # show ID if JSON payload, else show "QR"
                qid = None
                if isinstance(cache.get("payload"), dict):
                    qid = cache["payload"].get("id")
                if qid:
                    label = f"#{tid} {t['label']} QR:{qid}"
                else:
                    label = f"#{tid} {t['label']} QR"

            draw_bbox(annotated, t["bbox"], label, t["conf"])

        # 5) Zone counts (use tracked objects for stability)
        counts = {z["zone_id"]: 0 for z in self.zones}
        for t in tracks_out:
            zid = t.get("zone_id")
            if zid in counts:
                counts[zid] += 1
        debug["counts"] = counts

        # print("[DEBUG] [CV] ", datetime.now(timezone.utc).isoformat(), " active tracks:", [(t["track_id"], t["label"], t.get("zone_id")) for t in tracks_out])

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
            # 1) Publish TRACK-LEVEL events (best signal)
            for e in enters:
                try:
                    hinted_id, qr_meta = self._qr_meta_for_track(e["track_id"])
                    meta = {
                        "source": "phase2",
                        "mode": "enter",
                        "label": e["label"],
                        "track_id": e["track_id"],
                        "reason": e.get("reason"),
                        **qr_meta,
                    }
                    resp = self.publisher.publish_zone_change(
                        object_type=self.object_type,
                        from_zone=None,
                        to_zone=e["to_zone"],
                        hinted_object_id=None,
                        confidence=0.6,
                        meta=meta,
                    )
                    debug["published"].append(resp)
                except Exception as ex:
                    debug["published"].append({"error": str(ex), "event": e})

            for x in exits:
                try:
                    hinted_id, qr_meta = self._qr_meta_for_track(e["track_id"])
                    meta = {
                        "source": "phase2",
                        "mode": "enter",
                        "label": e["label"],
                        "track_id": e["track_id"],
                        "reason": e.get("reason"),
                        **qr_meta,
                    }
                    resp = self.publisher.publish_zone_change(
                        object_type=self.object_type,
                        from_zone=x["from_zone"],
                        to_zone=None,
                        hinted_object_id=None,
                        confidence=0.6,
                        meta=meta,
                    )
                    debug["published"].append(resp)
                except Exception as ex:
                    debug["published"].append({"error": str(ex), "event": x})

            for t in transfers:
                try:
                    hinted_id, qr_meta = self._qr_meta_for_track(e["track_id"])
                    meta = {
                        "source": "phase2",
                        "mode": "enter",
                        "label": e["label"],
                        "track_id": e["track_id"],
                        "reason": e.get("reason"),
                        **qr_meta,
                    }
                    resp = self.publisher.publish_zone_change(
                        object_type=self.object_type,
                        from_zone=t["from_zone"],
                        to_zone=t["to_zone"],
                        hinted_object_id=None,
                        confidence=0.7,
                        meta=meta,
                    )
                    debug["published"].append(resp)
                except Exception as ex:
                    debug["published"].append({"error": str(ex), "event": t})

            # 2) (Optional) Publish ZONE-LEVEL residual events (noisy, keep for analytics/debug)
            # Comment this out if it spams your backend.
            for r in residual:
                try:
                    resp = self.publisher.publish_zone_change(
                        object_type=self.object_type,
                        from_zone=r["from_zone"],
                        to_zone=r["to_zone"],
                        hinted_object_id=None,
                        confidence=0.5,
                        meta={
                            "source": "phase2",
                            "mode": r["mode"],   # "appearance" or "disappearance"
                            "old": r["old"],
                            "new": r["new"],
                        },
                    )
                    debug["published"].append(resp)
                except Exception as ex:
                    debug["published"].append({"error": str(ex), "event": r})

        else:
            # Standalone mode: print the events
            for t in transfers:
                for e in enters:
                    print(f"[CV] ENTER  #{e['track_id']} {e['label']} -> {e['to_zone']} ({e.get('reason')})")
                for x in exits:
                    print(f"[CV] EXIT   #{x['track_id']} {x['label']} {x['from_zone']} -> OUTSIDE ({x.get('reason')})")
                for t in transfers:
                    print(f"[CV] TRANSFER #{t['track_id']} {t['label']} {t['from_zone']} -> {t['to_zone']} ({t.get('reason')})")
                
                hinted_id, _ = self._qr_meta_for_track(t["track_id"])
                qr_txt = f" QR:{hinted_id}" if hinted_id else ""
                print(f"[CV] TRANSFER #{t['track_id']} {t['label']}{qr_txt} {t['from_zone']} -> {t['to_zone']} ({t.get('reason')})")
            
            for r in residual:
                if r["mode"] == "appearance":
                    print(f"[CV] APPEAR {r['to_zone']} ({r['old']} -> {r['new']})")
                else:
                    print(f"[CV] DISAPPEAR {r['from_zone']} ({r['old']} -> {r['new']})")

        return annotated, debug
    
    def _qr_meta_for_track(self, track_id: int):
        cache = self.track_qr_cache.get(track_id)
        if not cache:
            return None, {}

        payload = cache.get("payload")
        raw = cache.get("raw")

        hinted_id = None
        meta = {"qr_raw": raw}

        if isinstance(payload, dict):
            hinted_id = payload.get("id")  # we expect {"id":"PRUSA-01", ...}
            meta["qr_payload"] = payload

        return hinted_id, meta