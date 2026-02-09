from typing import Dict, List, Optional, Tuple
import math
import time

def bbox_center(b):
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

class SimpleTracker:
    """
    Lightweight multi-object tracker using centroid matching.
    - Maintains track IDs across frames
    - Emits transfers when a track changes zone
    - Handles multiple objects moving at once
    """
    def __init__(self, max_age_frames: int = 15, match_dist_px: float = 80.0):
        self.next_id = 1
        self.tracks: Dict[int, dict] = {}
        self.max_age_frames = int(max_age_frames)
        self.match_dist_px = float(match_dist_px)
        self.frame_i = 0

    def update(self, detections: List[dict]) -> Tuple[List[dict], List[dict]]:
        """
        detections: list of dicts with bbox, label, conf, zone_id
        Returns:
          - tracks_out: list of dicts {track_id, label, conf, bbox, zone_id, prev_zone_id}
          - transfers: list of dicts {track_id, label, from_zone, to_zone}
        """
        self.frame_i += 1
        now = time.time()

        # Prepare detection centers
        det_centers = []
        for d in detections:
            cx, cy = bbox_center(d["bbox"])
            det_centers.append((cx, cy))

        # Mark tracks as unmatched initially
        for tid in self.tracks:
            self.tracks[tid]["matched"] = False

        # Match detections to existing tracks (greedy nearest)
        assigned_track = [-1] * len(detections)
        used_tracks = set()

        for di, d in enumerate(detections):
            best_tid = None
            best_dist = 1e9
            cx, cy = det_centers[di]

            for tid, t in self.tracks.items():
                if tid in used_tracks:
                    continue
                # Option: enforce same label to reduce ID swaps
                if t["label"] != d["label"]:
                    continue
                tcx, tcy = t["center"]
                dd = dist((cx, cy), (tcx, tcy))
                if dd < best_dist and dd <= self.match_dist_px:
                    best_dist = dd
                    best_tid = tid

            if best_tid is not None:
                assigned_track[di] = best_tid
                used_tracks.add(best_tid)
                self.tracks[best_tid]["matched"] = True

        transfers = []
        tracks_out = []

        # Update matched tracks / create new tracks
        for di, d in enumerate(detections):
            cx, cy = det_centers[di]
            tid = assigned_track[di]

            if tid == -1:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "label": d["label"],
                    "center": (cx, cy),
                    "bbox": d["bbox"],
                    "conf": d["conf"],
                    "zone_id": d.get("zone_id"),
                    "prev_zone_id": None,
                    "last_seen_frame": self.frame_i,
                    "last_seen_time": now,
                    "matched": True,
                }
            else:
                t = self.tracks[tid]
                prev_zone = t.get("zone_id")
                new_zone = d.get("zone_id")

                # Save prev zone before overwriting
                t["prev_zone_id"] = prev_zone
                t["zone_id"] = new_zone
                t["center"] = (cx, cy)
                t["bbox"] = d["bbox"]
                t["conf"] = d["conf"]
                t["last_seen_frame"] = self.frame_i
                t["last_seen_time"] = now

                # Transfer event if zone changed and both are real zones
                if prev_zone != new_zone and prev_zone is not None and new_zone is not None:
                    transfers.append({
                        "track_id": tid,
                        "label": t["label"],
                        "from_zone": prev_zone,
                        "to_zone": new_zone
                    })

            t = self.tracks[tid]
            tracks_out.append({
                "track_id": tid,
                "label": t["label"],
                "conf": t["conf"],
                "bbox": t["bbox"],
                "zone_id": t.get("zone_id"),
                "prev_zone_id": t.get("prev_zone_id")
            })

        # Remove old tracks not seen recently
        to_delete = []
        for tid, t in self.tracks.items():
            age = self.frame_i - t["last_seen_frame"]
            if age > self.max_age_frames:
                to_delete.append(tid)
        for tid in to_delete:
            self.tracks.pop(tid, None)

        return tracks_out, transfers