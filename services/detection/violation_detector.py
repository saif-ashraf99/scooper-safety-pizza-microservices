from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from services.shared import Detection, Violation, ViolationType, DetectionClass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ViolationDetector:
    """
    Emit ONE `NO_SCOOPER` violation per continuous episode.
    A new episode can start only after the ROI has been “safe”
    for `frames_clear` consecutive frames.
    """

    def __init__(self, rois: List[Dict[str, Any]], *, frames_clear: int = 12):
        """
        Args
        ----
        rois : list of ROI dicts (each needs 'id', 'name', 'coordinates')
        frames_clear : how many consecutive safe frames must pass
                       before the ROI can fire again (default 12 ≈ 0.4 s at 30 FPS)
        """
        self.rois = rois
        self.tracker = DeepSort(max_age=30, n_init=0)   # confirm tracks immediately
        self.scooper_threshold = 0.7
        self.frames_clear = frames_clear

        # per-ROI episode state
        # { roi_id: {"violating": bool, "cooldown": int} }
        self.roi_state: Dict[str, Dict[str, int | bool]] = {
            roi["id"]: {"violating": False, "cooldown": 0} for roi in rois
        }

    # ---------------------- helpers ----------------------
    @staticmethod
    def center(b: List[float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = b
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def iou_inside(b: "BBox", roi_xyxy: Tuple[int, int, int, int]) -> bool:
        """True iff the two boxes intersect (PERSON fallback)."""
        x1, y1, x2, y2 = roi_xyxy
        return not (b.x2 < x1 or b.x1 > x2 or b.y2 < y1 or b.y1 > y2)

    # ---------------------- main -------------------------
    def detect_violations(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_id: str,
    ) -> List[Violation]:

        # 1) --------- track HAND / PERSON detections -------------------------
        trackables = [
            d for d in detections
            if d.class_name in (DetectionClass.HAND, DetectionClass.PERSON)
        ]
        inputs = [
            ([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2], d.confidence, d.class_name)
            for d in trackables
        ]
        tracks = self.tracker.update_tracks(inputs, frame=frame)

        # push track IDs back onto detections (nearest-centroid match)
        det_centroids = [self.center([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
                         for d in trackables]
        for trk in tracks:
            tid = trk.track_id
            l, t, r, b = trk.to_ltrb()
            cx, cy = (l + r) / 2, (t + b) / 2
            if det_centroids:
                idx = int(np.argmin([np.hypot(cx - x, cy - y) for x, y in det_centroids]))
                trackables[idx].track_id = tid

        # 2) --------- helpers ------------------------------------------------
        scoopers = [d for d in detections if d.class_name == DetectionClass.SCOOPER]

        hands = [d for d in detections if d.class_name == DetectionClass.HAND]
        using_person = False
        if not hands:
            hands = [d for d in detections if d.class_name == DetectionClass.PERSON]
            using_person = True

        violations: List[Violation] = []

        # 3) --------- evaluate each ROI once ---------------------------------
        for roi in self.rois:
            rid = roi["id"]
            if not roi.get("active", True):
                continue

            state = self.roi_state.setdefault(rid, {"violating": False, "cooldown": 0})

            x1, y1, x2, y2 = roi["coordinates"]
            w = x2 - x1
            max_dist = w * 0.20
            violating_now = False
            violator_det = None

            for h in hands:
                # inside?
                in_roi = (
                    x1 <= (h.bbox.x1 + h.bbox.x2) / 2 <= x2
                    and y1 <= (h.bbox.y1 + h.bbox.y2) / 2 <= y2
                ) if not using_person else self.iou_inside(h.bbox, (x1, y1, x2, y2))

                if not in_roi:
                    continue

                # nearby scooper?
                cx, cy = self.center([h.bbox.x1, h.bbox.y1, h.bbox.x2, h.bbox.y2])
                scooper_near = any(
                    np.hypot(cx - sx, cy - sy) < max_dist and s.confidence > self.scooper_threshold
                    for s in scoopers
                    for sx, sy in [self.center([s.bbox.x1, s.bbox.y1, s.bbox.x2, s.bbox.y2])]
                )

                if not scooper_near:
                    violating_now = True
                    violator_det = h
                    break

            # 4) --------- state machine --------------------------------------
            if violating_now:
                # ROI is unsafe this frame
                if not state["violating"] and state["cooldown"] == 0:
                    # NEW episode: emit exactly once
                    desc = f"Hand/person in {roi['name']} without scooper"
                    violations.append(
                        Violation(
                            type=ViolationType.NO_SCOOPER,
                            roi_id=rid,
                            confidence=violator_det.confidence,
                            bbox=violator_det.bbox,
                            description=desc,
                            track_id=getattr(violator_det, "track_id", None),
                        )
                    )
                    logger.warning(f"[{frame_id}] ROI {rid} VIOLATION: {desc}")
                state["violating"] = True
                state["cooldown"] = 0  # hold off resetting until safe
            else:
                # ROI safe this frame
                if state["violating"]:
                    # just transitioned out of violation → start cooldown
                    state["violating"] = False
                    state["cooldown"] = self.frames_clear
                elif state["cooldown"] > 0:
                    state["cooldown"] -= 1

        return violations
