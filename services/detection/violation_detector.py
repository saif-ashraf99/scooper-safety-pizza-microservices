import numpy as np
import logging
from typing import List, Dict, Any, Tuple

from services.shared import Detection, Violation, ViolationType, DetectionClass

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ViolationDetector:
    """Violation detection logic"""
    
    def __init__(self, rois: List[Dict[str, Any]]):
        self.rois = rois
        self.hand_history = {}  # Track hand movements
        self.scooper_threshold = 0.7
        
    def is_point_in_roi(self, point: Tuple[float, float], roi: Dict[str, Any]) -> bool:
        """Check if point is inside ROI"""
        x, y = point
        x1, y1, x2, y2 = roi['coordinates']
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    

    def detect_violations(
        self,
        detections: List[Detection],
        frame_id: str
    ) -> List[Violation]:
        """No‐scooper when HAND (or PERSON) enters the ROI rectangle at all."""
        violations: List[Violation] = []

        # 1) scooper detections
        scoopers = [d for d in detections if d.class_name == DetectionClass.SCOOPER]

        # 2) hand detections, fallback to person if none
        hands = [d for d in detections if d.class_name == DetectionClass.HAND]
        using_person = False
        if not hands:
            hands = [d for d in detections if d.class_name == DetectionClass.PERSON]
            using_person = True
            logger.debug(f"[{frame_id}] no HANDs → falling back to {len(hands)} PERSON(s)")

        # 3) check each “hand” vs each ROI
        for hand in hands:
            # get the raw bbox coords
            bx1, by1, bx2, by2 = hand.bbox.x1, hand.bbox.y1, hand.bbox.x2, hand.bbox.y2

            for roi in self.rois:
                if not roi.get("active", True):
                    continue

                rx1, ry1, rx2, ry2 = roi["coordinates"]

                # if we have a true hand, just test its center
                if not using_person:
                    cx, cy = (bx1+bx2)/2, (by1+by2)/2
                    inside = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
                    logger.debug(f"[{frame_id}] hand‐center at {(cx,cy)} inside ROI? {inside}")
                else:
                    # PERSON fallback: test ANY intersection of boxes
                    inter = not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2)
                    inside = inter
                    logger.debug(f"[{frame_id}] person‐bbox {(bx1,by1,bx2,by2)} "
                                f"intersects ROI {(rx1,ry1,rx2,ry2)}? {inside}")

                if not inside:
                    continue

                # 4) compute a dynamic distance threshold
                max_dist = (rx2 - rx1) * 0.2

                # 5) look for a nearby scooper
                scooper_present = False
                for s in scoopers:
                    sx, sy = (s.bbox.x1 + s.bbox.x2)/2, (s.bbox.y1 + s.bbox.y2)/2
                    dist = np.hypot(cx - sx, cy - sy) if not using_person else \
                        np.hypot((by2+by1)/2 - sy, sx - (bx1+bx2)/2)
                    logger.debug(f"[{frame_id}] scooper @ {(sx,sy)}, dist={dist:.1f}")
                    if dist < max_dist and s.confidence > self.scooper_threshold:
                        scooper_present = True
                        break

                if not scooper_present:
                    desc = f"Hand/person in {roi['name']} without scooper"
                    v = Violation(
                        type=ViolationType.NO_SCOOPER,
                        roi_id=roi["id"],
                        confidence=hand.confidence,
                        bbox=hand.bbox,
                        description=desc
                    )
                    violations.append(v)
                    logger.warning(f"[{frame_id}] VIOLATION: {desc}")

        return violations
