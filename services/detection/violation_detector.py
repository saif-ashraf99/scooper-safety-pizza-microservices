from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from services.shared import Detection, Violation, ViolationType, DetectionClass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ViolationDetector:
    def __init__(self, rois: List[Dict[str, Any]]):
        self.rois = rois
        # confirm new detections immediately
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.scooper_threshold = 0.7
        # remember which (track_id, roi_id) combos have already violated
        self.active_violations = set()

    def get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def detect_violations(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_id: str
    ) -> List[Violation]:
        """Track hands/people with DeepSORT, then emit at most one violation per hand+ROI."""
        # --- 1) isolate hands/persons and run DeepSORT ---
        track_dets = [d for d in detections if d.class_name in (DetectionClass.HAND, DetectionClass.PERSON)]
        deep_inputs = [
            ([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2], d.confidence, d.class_name)
            for d in track_dets
        ]
        tracks = self.tracker.update_tracks(deep_inputs, frame=frame)
        logger.debug(f"[{frame_id}] DeepSORT returned {len(tracks)} tracks")

        # --- 2) centroid‐based matching of track → detection ---
        det_centroids = [
            ((d.bbox.x1 + d.bbox.x2) / 2, (d.bbox.y1 + d.bbox.y2) / 2)
            for d in track_dets
        ]
        for track in tracks:
            tid = track.track_id
            l, t, r, b = track.to_ltrb()
            cx_t, cy_t = (l + r) / 2, (t + b) / 2
            if not det_centroids:
                continue
            # find nearest detection index
            dists = [np.hypot(cx_t - cx, cy_t - cy) for cx, cy in det_centroids]
            best_idx = int(np.argmin(dists))
            # assign unconditionally to ensure every track_det gets an ID
            track_dets[best_idx].track_id = tid
            logger.debug(f"[{frame_id}] assigned track {tid} to track_det #{best_idx}")

        # --- 3) look for scooper violations, but only new ones ---
        violations: List[Violation] = []
        scoopers = [d for d in detections if d.class_name == DetectionClass.SCOOPER]

        # HANDs first, fallback to PERSON
        hands = [d for d in detections if d.class_name == DetectionClass.HAND]
        using_person = False
        if not hands:
            hands = [d for d in detections if d.class_name == DetectionClass.PERSON]
            using_person = True
            logger.debug(f"[{frame_id}] no HANDs → falling back to {len(hands)} PERSON(s)")

        # track which violations are still present this frame
        still_active = set()

        for hand in hands:
            track_id = getattr(hand, 'track_id', None)
            cx, cy = self.get_bbox_center([hand.bbox.x1, hand.bbox.y1, hand.bbox.x2, hand.bbox.y2])

            for roi in self.rois:
                if not roi.get("active", True):
                    continue
                x1, y1, x2, y2 = roi["coordinates"]
                inside = (
                    (x1 <= cx <= x2 and y1 <= cy <= y2)
                    if not using_person
                    else not (
                        hand.bbox.x2 < x1 or hand.bbox.x1 > x2 or
                        hand.bbox.y2 < y1 or hand.bbox.y1 > y2
                    )
                )
                if not inside:
                    continue

                key = (track_id, roi["id"])
                still_active.add(key)

                # check for a nearby scooper
                max_dist = (x2 - x1) * 0.2
                scooper_present = False
                for s in scoopers:
                    sx, sy = self.get_bbox_center([s.bbox.x1, s.bbox.y1, s.bbox.x2, s.bbox.y2])
                    if np.hypot(cx - sx, cy - sy) < max_dist and s.confidence > self.scooper_threshold:
                        scooper_present = True
                        break

                # emit only if newly violated
                if not scooper_present and key not in self.active_violations:
                    self.active_violations.add(key)
                    desc = f"Hand/person in {roi['name']} without scooper"
                    v = Violation(
                        type=ViolationType.NO_SCOOPER,
                        roi_id=roi["id"],
                        confidence=hand.confidence,
                        bbox=hand.bbox,
                        description=desc,
                        track_id=track_id
                    )
                    violations.append(v)
                    logger.warning(f"[{frame_id}] track {track_id} VIOLATION: {desc}")

        # --- 4) prune any tracked violations that no longer apply ---
        self.active_violations &= still_active

        return violations
