import json
import yaml
import time

from cv.detectors.yolo_detector import YOLODetector
from cv.tracking.zone_mapper import assign_to_zones
from cv.tracking.state_tracker import ZoneStateTracker
from cv.utils.draw import draw_rect_zone, draw_bbox
from cv.tracking.simple_tracker import SimpleTracker
from cv.qr.qr_reader import QRReader
# from cv.color.color_detector import FilamentColorDetector
from datetime import datetime, timezone
from cv.events.rabbitmq_publisher import RabbitMQPublisher


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

        print("[CV] YOLO model:", cfg["yolo"]["model"])

        self.class_filter = set(cfg.get("detect_classes") or [])
        self.process_every_n = int(cfg["logic"]["process_every_n_frames"])
        # self.publish_events = bool(cfg["logic"]["publish_events"])
        self.object_type = "filament_spool"

        self.min_unknown_frames = int(cfg["logic"].get("min_unknown_frames", 6))
        self._unknown_age = {}  # track_id -> {"zone_id": str, "age": int}

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
        self.qr_full_frame = True           # use full-frame decoding
        self.qr_assign_max_px = 220         # tune for your camera distance
        self.qr_full_every_n = 2            # decode full frame every N processed frames

        # cache last known QR per track so you don't need to decode every frame
        self.track_qr_cache = {}  # track_id -> {"raw": str, "payload": dict}

        # self.color_detector = FilamentColorDetector()
        # self.track_color_cache = {}  # track_id -> {"color": str, "conf": float}
        # self.color_every_n = 3       # compute color every N processed frames

        # Publisher is optional, and imported only if needed
        # self.publisher = None
        # if self.publish_events:
        #     from cv.events.event_publisher import EventPublisher  # lazy import
        #     self.publisher = EventPublisher(
        #         base_url=cfg["backend"]["base_url"],
        #         path=cfg["backend"]["cv_event_path"],
        #         timeout_seconds=int(cfg["backend"]["timeout_seconds"]),
        #     )

        # RabbitMQ publisher
        rmq = cfg.get("rabbitmq", {}) or {}
        self.rmq_enabled = bool(rmq.get("enabled", False))
        self.publisher_rmq = None
        if self.rmq_enabled:
            self.publisher_rmq = RabbitMQPublisher(
                host=rmq.get("host", "206.180.209.81"),
                port=int(rmq.get("port", 5672)),
                username=rmq.get("username", "twin"),
                password=rmq.get("password", "twinrabbitpass"),
                vhost=rmq.get("vhost", "/"),
                exchange=rmq.get("exchange", "cv.events"),
            )

        self.camera_id = cfg.get("camera_id", "cam_1")

        # Anomaly detector (COCO) for "other objects"
        anom = cfg.get("anomaly", {}) or {}
        self.anomaly_enabled = bool(anom.get("enabled", False))
        self.anomaly_every_n = int(anom.get("every_n_frames", 10))
        self.anomaly_ignore = set(anom.get("ignore_labels", ["person"]) or [])
        self.anomaly_detector = None
        if self.anomaly_enabled:
            self.anomaly_detector = YOLODetector(
                model_path=anom.get("model", "yolov8n.pt"),
                conf=float(anom.get("conf", 0.25)),
                iou=float(anom.get("iou", 0.45)),
                device=str(anom.get("device", "cpu")),
            )

        # publish only when changed to reduce spam
        self._last_inventory_payload = None
        self._last_anomaly_payload = None
        self.publish_min_interval_sec = 0.5
        self._last_publish_ts = 0.0

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
        dets = [d for d in dets if d.get("label") != "person"]

        # Debug: show what YOLO sees
        # if dets:
        #     print("[YOLO] detections:", [(d["label"], round(d["conf"], 2)) for d in dets])
        # else:
        #     print("[YOLO] detections: []")

        # 2) Filter + zone-assign
        dets = self._filter_dets(dets)
        dets = [d for d in dets if d["conf"] >= 0.25]
        dets = assign_to_zones(dets, self.zones)

        # 3) Tracking-based transfers (best for MOVE events)
        tracks_out, transfers, enters, exits = self.tracker.update(dets)
        # print("[DEBUG tracks]", [(t["track_id"], t["label"], t.get("prev_zone_id"), t.get("zone_id")) for t in tracks_out])
        # print("[DEBUG events]", "T=", len(transfers), "E=", len(enters), "X=", len(exits))
        # --- Color estimation (cached) ---
        # if (self.frame_i % self.color_every_n) == 0:
        #     for t in tracks_out:
        #         tid = t["track_id"]
        #         x1, y1, x2, y2 = t["bbox"]
        #         crop = frame_bgr[y1:y2, x1:x2]

        #         color, cconf = self.color_detector.classify(crop)

        #         prev = self.track_color_cache.get(tid)
        #         # overwrite if better confidence, or if no previous
        #         if (prev is None) or (cconf >= (prev.get("conf", 0.0) + 0.05)):
        #             self.track_color_cache[tid] = {"color": color, "conf": cconf}

        # --- QR decode (full-frame) and assign to nearest spool track ---
        if self.qr_enabled and self.qr_reader is not None and (self.frame_i % self.qr_full_every_n == 0):
            qr_list = self.qr_reader.decode_multi_bgr(frame_bgr)

            # precompute track centers
            track_centers = []
            for t in tracks_out:
                x1, y1, x2, y2 = t["bbox"]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                track_centers.append((t["track_id"], cx, cy))

            for qr in qr_list:
                payload = qr.get("payload")
                if not isinstance(payload, dict):
                    continue
                spool_id = payload.get("id")
                if not spool_id:
                    continue

                # if multi decode gave center, match by nearest distance
                if qr.get("center") and track_centers:
                    qx, qy = qr["center"]
                    best_tid = None
                    best_d2 = None
                    for tid, cx, cy in track_centers:
                        dx, dy = (cx - qx), (cy - qy)
                        d2 = dx*dx + dy*dy
                        if best_d2 is None or d2 < best_d2:
                            best_d2 = d2
                            best_tid = tid

                    if best_tid is not None and best_d2 is not None:
                        if best_d2 <= (self.qr_assign_max_px * self.qr_assign_max_px):
                            self.track_qr_cache[best_tid] = {"raw": qr["raw"], "payload": payload}
                else:
                    # fallback: no center info; can't match robustly
                    # (ignore or set global cache)
                    pass

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

            # col = self.track_color_cache.get(tid, {}).get("color")
            # if col and col != "unknown":
            #     label = f"{label} [{col}]"

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
        # if self.publish_events and self.publisher is not None:
            # 1) Publish TRACK-LEVEL events (best signal)
            # for e in enters:
            #     try:
            #         hinted_id, qr_meta = self._qr_meta_for_track(e["track_id"])
            #         col = self.track_color_cache.get(e["track_id"], {}).get("color", "unknown")
            #         ccf = self.track_color_cache.get(e["track_id"], {}).get("conf", 0.0)
            #         meta = {
            #             "source": "phase2",
            #             "mode": "enter",
            #             "label": e["label"],
            #             "track_id": e["track_id"],
            #             "reason": e.get("reason"),
            #             "filament_color": col,
            #             "filament_color_conf": ccf,
            #             **qr_meta,
            #         }
            #         resp = self.publisher.publish_zone_change(
            #             object_type=self.object_type,
            #             from_zone=None,
            #             to_zone=e["to_zone"],
            #             hinted_object_id=hinted_id,
            #             confidence=0.6,
            #             meta=meta,
            #         )
            #         debug["published"].append(resp)
            #     except Exception as ex:
            #         debug["published"].append({"error": str(ex), "event": e})

            # for x in exits:
            #     try:
            #         hinted_id, qr_meta = self._qr_meta_for_track(x["track_id"])
            #         col = self.track_color_cache.get(x["track_id"], {}).get("color", "unknown")
            #         ccf = self.track_color_cache.get(x["track_id"], {}).get("conf", 0.0)
            #         meta = {
            #             "source": "phase2",
            #             "mode": "exit",
            #             "label": x["label"],
            #             "track_id": x["track_id"],
            #             "reason": x.get("reason"),
            #             "filament_color": col,
            #             "filament_color_conf": ccf,
            #             **qr_meta,
            #         }
            #         resp = self.publisher.publish_zone_change(
            #             object_type=self.object_type,
            #             from_zone=x["from_zone"],
            #             to_zone=None,
            #             hinted_object_id=hinted_id,
            #             confidence=0.6,
            #             meta=meta,
            #         )
            #         debug["published"].append(resp)
            #     except Exception as ex:
            #         debug["published"].append({"error": str(ex), "event": x})

            # for t in transfers:
            #     try:
            #         hinted_id, qr_meta = self._qr_meta_for_track(t["track_id"])
            #         col = self.track_color_cache.get(t["track_id"], {}).get("color", "unknown")
            #         ccf = self.track_color_cache.get(t["track_id"], {}).get("conf", 0.0)
            #         meta = {
            #             "source": "phase2",
            #             "mode": "transfer",
            #             "label": t["label"],
            #             "track_id": t["track_id"],
            #             "reason": t.get("reason"),
            #             "filament_color": col,
            #             "filament_color_conf": ccf,
            #             **qr_meta,
            #         }
            #         resp = self.publisher.publish_zone_change(
            #             object_type=self.object_type,
            #             from_zone=t["from_zone"],
            #             to_zone=t["to_zone"],
            #             hinted_object_id=hinted_id,
            #             confidence=0.7,
            #             meta=meta,
            #         )
            #         debug["published"].append(resp)
            #     except Exception as ex:
            #         debug["published"].append({"error": str(ex), "event": t})

            # # 2) (Optional) Publish ZONE-LEVEL residual events (noisy, keep for analytics/debug)
            # # Comment this out if it spams your backend.
            # for r in residual:
            #     try:
            #         resp = self.publisher.publish_zone_change(
            #             object_type=self.object_type,
            #             from_zone=r["from_zone"],
            #             to_zone=r["to_zone"],
            #             hinted_object_id=None,
            #             confidence=0.5,
            #             meta={
            #                 "source": "phase2",
            #                 "mode": r["mode"],   # "appearance" or "disappearance"
            #                 "old": r["old"],
            #                 "new": r["new"],
            #             },
            #         )
            #         debug["published"].append(resp)
            #     except Exception as ex:
            #         debug["published"].append({"error": str(ex), "event": r})

        
        # -------------------------------
        # Debug prints (only when useful)
        # -------------------------------
        if enters or exits or transfers or residual:
            for e in enters:
                hinted_id, _ = self._qr_meta_for_track(e["track_id"])
                qr_txt = f" QR:{hinted_id}" if hinted_id else ""
                print(f"[CV] ENTER    #{e['track_id']} {e['label']}{qr_txt} -> {e['to_zone']} ({e.get('reason')})")

            for x in exits:
                hinted_id, _ = self._qr_meta_for_track(x["track_id"])
                qr_txt = f" QR:{hinted_id}" if hinted_id else ""
                print(f"[CV] EXIT     #{x['track_id']} {x['label']}{qr_txt} {x['from_zone']} -> OUTSIDE ({x.get('reason')})")

            for t in transfers:
                hinted_id, _ = self._qr_meta_for_track(t["track_id"])
                qr_txt = f" QR:{hinted_id}" if hinted_id else ""
                print(f"[CV] TRANSFER #{t['track_id']} {t['label']}{qr_txt} {t['from_zone']} -> {t['to_zone']} ({t.get('reason')})")

            for r in residual:
                if r["mode"] == "appearance":
                    print(f"[CV] APPEAR      {r['to_zone']} ({r['old']} -> {r['new']})")
                else:
                    print(f"[CV] DISAPPEAR   {r['from_zone']} ({r['old']} -> {r['new']})")


        # ---------------------------------------
        # Build payloads and print before publish
        # ---------------------------------------
        inv_payload = None
        anom_payload = None

        if self.rmq_enabled and self.publisher_rmq is not None:
            inv_payload = self._build_zone_inventory_payload(tracks_out)

            # Print inventory payload only when it changes
            if inv_payload != self._last_inventory_payload:
                print("[RMQ] would publish cv.zone.inventory:", inv_payload)

            # anomaly payload only every N frames
            if self.anomaly_enabled and (self.frame_i % self.anomaly_every_n == 0):
                anom_payload = self._build_zone_anomaly_payload(frame_bgr,counts)
                if anom_payload != self._last_anomaly_payload:
                    print("[RMQ] would publish cv.zone.anomaly:", anom_payload)

            # ---------------------------------------
            # Actually publish (still change-throttled)
            # ---------------------------------------
            self._maybe_publish("cv.zone.inventory", inv_payload, kind="inventory")

            if anom_payload is not None:
                self._maybe_publish("cv.zone.anomaly", anom_payload, kind="anomaly")

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
    
    def _now_iso(self):
        return datetime.now(timezone.utc).isoformat()

    def _get_spool_id_for_track(self, track_id: int):
        hinted_id, _ = self._qr_meta_for_track(track_id)
        return hinted_id  # QR id == spool_id (recommended)

    def _build_zone_inventory_payload(self, tracks_out):
        zones = {z["zone_id"]: {"spool_ids": [], "unknown_spool_count": 0} for z in self.zones}

        seen_track_ids = set()

        for t in tracks_out:
            tid = t["track_id"]
            zid = t.get("zone_id")
            if zid not in zones:
                continue

            seen_track_ids.add(tid)

            spool_id = self._get_spool_id_for_track(tid)

            if spool_id:
                # QR-confirmed: count immediately + clear unknown age
                zones[zid]["spool_ids"].append(spool_id)
                if tid in self._unknown_age:
                    del self._unknown_age[tid]
            else:
                # Unknown spool: require it to persist for min_unknown_frames
                prev = self._unknown_age.get(tid)

                if prev is None or prev["zone_id"] != zid:
                    self._unknown_age[tid] = {"zone_id": zid, "age": 1}
                else:
                    prev["age"] += 1

                if self._unknown_age[tid]["age"] >= self.min_unknown_frames:
                    zones[zid]["unknown_spool_count"] += 1

        # Cleanup stale tracks (not seen anymore)
        stale = [tid for tid in self._unknown_age.keys() if tid not in seen_track_ids]
        for tid in stale:
            del self._unknown_age[tid]

        # stabilize spool_ids for comparison
        for zid in zones:
            zones[zid]["spool_ids"] = sorted(list(set(zones[zid]["spool_ids"])))

        return {
            "type": "zone.inventory",
            "ts": self._now_iso(),
            "camera_id": self.camera_id,
            "zones": zones
        }

    def _build_zone_anomaly_payload(self, frame_bgr, spool_counts: dict):
        """
        other_count = coco_count - spool_count (per zone)
        spool_counts: {zone_id: int} from your tracked spool counts
        """
        zones = {z["zone_id"]: {"other_count": 0} for z in self.zones}

        if not self.anomaly_detector:
            return {
                "type": "zone.anomaly",
                "ts": self._now_iso(),
                "camera_id": self.camera_id,
                "zones": zones
            }

        dets = self.anomaly_detector.detect(frame_bgr)
        dets = [d for d in dets if d.get("label") not in self.anomaly_ignore]
        dets = [d for d in dets if d.get("conf", 0.0) >= 0.25]
        dets = assign_to_zones(dets, self.zones)

        # count COCO objects per zone
        coco_counts = {z["zone_id"]: 0 for z in self.zones}
        for d in dets:
            zid = d.get("zone_id")
            if zid in coco_counts:
                coco_counts[zid] += 1

        # compute other_count = max(0, coco - spool)
        for zid in zones:
            spool_c = int(spool_counts.get(zid, 0))
            coco_c = int(coco_counts.get(zid, 0))
            zones[zid]["other_count"] = max(0, coco_c - spool_c)

        print("[ANOM] coco_counts:", coco_counts, "spool_counts:", spool_counts, "other:", zones)

        return {
            "type": "zone.anomaly",
            "ts": self._now_iso(),
            "camera_id": self.camera_id,
            "zones": zones
        }

    def _maybe_publish(self, routing_key: str, payload: dict, kind: str):
        # publish only when changed, and at least every publish_min_interval_sec
        now = time.time()
        if (now - self._last_publish_ts) < self.publish_min_interval_sec:
            return

        last = self._last_inventory_payload if kind == "inventory" else self._last_anomaly_payload
        if payload == last:
            return

        if self.publisher_rmq:
            self.publisher_rmq.publish(routing_key, payload)

        if kind == "inventory":
            self._last_inventory_payload = payload
        else:
            self._last_anomaly_payload = payload

        self._last_publish_ts = now