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
        self.tracker = DeepSort(max_age=30, n_init=0)
        self.scooper_threshold = 0.7
        self.frames_clear = frames_clear

        # per-ROI episode state
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

    def _get_trackables(self, detections: List[Detection]) -> List[Detection]:
        return [
            d for d in detections
            if d.class_name in (DetectionClass.HAND, DetectionClass.PERSON)
        ]

    def _assign_track_ids(self, trackables: List[Detection], tracks: Any) -> None:
        det_centroids = [
            self.center([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2])
            for d in trackables
        ]
        for trk in tracks:
            tid = trk.track_id
            l, t, r, b = trk.to_ltrb()
            cx, cy = (l + r) / 2, (t + b) / 2
            if det_centroids:
                idx = int(np.argmin([
                    np.hypot(cx - x, cy - y) for x, y in det_centroids
                ]))
                trackables[idx].track_id = tid

    def _get_hands_and_mode(
        self, detections: List[Detection]
    ) -> Tuple[List[Detection], bool]:
        hands = [d for d in detections if d.class_name == DetectionClass.HAND]
        using_person = False
        if not hands:
            hands = [d for d in detections if d.class_name == DetectionClass.PERSON]
            using_person = True
        return hands, using_person

    def _evaluate_roi(
        self,
        roi: Dict[str, Any],
        hands: List[Detection],
        scoopers: List[Detection],
        using_person: bool,
        frame_id: str
    ) -> List[Violation]:
        rid = roi["id"]
        if not roi.get("active", True):
            return []

        state = self.roi_state.setdefault(rid, {"violating": False, "cooldown": 0})
        x1, y1, x2, y2 = roi["coordinates"]
        w = x2 - x1
        max_dist = w * 0.20
        violating_now = False
        violator_det = None

        for h in hands:
            if not using_person:
                cx, cy = (h.bbox.x1 + h.bbox.x2) / 2, (h.bbox.y1 + h.bbox.y2) / 2
                in_roi = x1 <= cx <= x2 and y1 <= cy <= y2
            else:
                in_roi = self.iou_inside(h.bbox, (x1, y1, x2, y2))

            if not in_roi:
                continue

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

        results: List[Violation] = []
        # state machine
        if violating_now:
            if not state["violating"] and state["cooldown"] == 0:
                desc = f"Hand/person in {roi['name']} without scooper"
                results.append(
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
            state["cooldown"] = 0
        else:
            if state["violating"]:
                state["violating"] = False
                state["cooldown"] = self.frames_clear
            elif state["cooldown"] > 0:
                state["cooldown"] -= 1

        return results

    # ---------------------- main -------------------------
    def detect_violations(
        self,
        detections: List[Detection],
        frame: np.ndarray,
        frame_id: str,
    ) -> List[Violation]:
        # track HAND / PERSON
        trackables = self._get_trackables(detections)
        inputs = [
            ([d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2], d.confidence, d.class_name)
            for d in trackables
        ]
        tracks = self.tracker.update_tracks(inputs, frame=frame)
        self._assign_track_ids(trackables, tracks)

        # filter scoopers and hands
        scoopers = [d for d in detections if d.class_name == DetectionClass.SCOOPER]
        hands, using_person = self._get_hands_and_mode(detections)

        violations: List[Violation] = []
        for roi in self.rois:
            violations.extend(
                self._evaluate_roi(roi, hands, scoopers, using_person, frame_id)
            )

        return violations
