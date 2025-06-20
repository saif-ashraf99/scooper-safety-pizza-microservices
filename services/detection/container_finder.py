import os
import cv2
import numpy as np
from typing import Optional, Tuple, List
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerFinder:
    def __init__(
        self,
        tpl_dir: Optional[str] = None,
        match_threshold: float = 0.6,
        use_edges: bool = False,
        blur_ksize: Tuple[int, int] = (5, 5),
        peak_threshold: float = 0.05,  # Minimum difference between best and second-best match
        template_matching_method: int = cv2.TM_CCOEFF_NORMED,
    ):
        # 1) locate the folder of patches
        base = os.path.dirname(__file__)
        imgs_folder = tpl_dir or os.path.join(base, "imgs")
        if not os.path.isdir(imgs_folder):
            raise FileNotFoundError(f"[ContainerFinder] no imgs folder at {imgs_folder}")

# 2) loa/*************  ✨ Windsurf Command ⭐  *************/
        """
        Initialize a ContainerFinder.

        Args:
            tpl_dir: The directory containing the image patches of the containers.
                Defaults to the "imgs" folder in the same directory as the ContainerFinder class.
            match_threshold: The minimum match similarity required for a detection.
                Defaults to 0.6.
            use_edges: If True, use edges of the templates and images instead of the original color.
                Defaults to False.
            blur_ksize: The size of the blur kernel to apply to the template and image.
                Defaults to (5, 5).
            multi_scale: If True, search for templates at multiple scales.
                Defaults to True.
            scale_range: The range of scale factors to search for templates.
                Defaults to (0.5, 1.5).
            scale_steps: The number of scale steps to search for templates.
                Defaults to 10.
            nms_threshold: The minimum IoU overlap between detected bounding boxes to be considered the same object.
                Defaults to 0.3.
/*******  9e69309f-a7e7-46eb-a8e2-cb8c0979c4be  *******/d every .png in there as a separate template
                Defaults to 30.
            adaptive_threshold: If True, adapt the threshold value based on the confidence of the matches.
                Defaults to True.
            use_multiple_methods: If True, use both template matching and edge matching methods.
                Defaults to True.
        """
        self.templates: List[np.ndarray] = []
        self.template_names: List[str] = []
        for fn in sorted(os.listdir(imgs_folder)):
            if fn.lower().endswith(".png"):
                path = os.path.join(imgs_folder, fn)
                tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tpl is None:
                    raise IOError(f"[ContainerFinder] failed to load {path}")
                self.templates.append(tpl)
                self.template_names.append(fn)
                logger.info(f"[ContainerFinder] loaded patch {fn} size={tpl.shape[::-1]}")
        
        if not self.templates:
            raise ValueError(f"[ContainerFinder] no .png templates found in {imgs_folder}")

        self.th = match_threshold
        self.use_edges = use_edges
        self.blur_ksize = blur_ksize
        self.peak_threshold = peak_threshold
        self.method = template_matching_method

        # Precompute edge versions if requested
        if self.use_edges:
            self.templates = [cv2.Canny(tpl, 50, 150) for tpl in self.templates]

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Simple but effective preprocessing"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Only blur if kernel size is reasonable
        if self.blur_ksize[0] > 1 and self.blur_ksize[1] > 1:
            gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        
        if self.use_edges:
            gray = cv2.Canny(gray, 50, 150)
        
        return gray

    def _is_valid_match(self, result: np.ndarray, max_val: float, max_loc: Tuple[int, int]) -> bool:
        """Check if a match is likely to be a true positive"""
        
        # 1) Basic threshold check
        if max_val < self.th:
            return False
        
        # 2) Peak prominence check - ensure the match is significantly better than surrounding area
        h, w = result.shape
        y, x = max_loc[1], max_loc[0]  # Note: max_loc is (x, y)
        
        # Define search area around the peak (avoid edges)
        margin = 10
        y1 = max(0, y - margin)
        y2 = min(h, y + margin)
        x1 = max(0, x - margin) 
        x2 = min(w, x + margin)
        
        if y2 - y1 < 5 or x2 - x1 < 5:  # Too close to edge
            return True  # Accept edge cases
        
        # Get local region and find second highest value
        local_region = result[y1:y2, x1:x2].copy()
        local_region[y-y1, x-x1] = -1  # Remove the peak itself
        second_best = np.max(local_region)
        
        peak_prominence = max_val - second_best
        
        if peak_prominence < self.peak_threshold:
            logger.debug(f"Match rejected: weak peak prominence {peak_prominence:.3f}")
            return False
        
        # 3) Check for multiple similar peaks (indicates pattern repetition/noise)
        similar_peaks = np.sum(result > (max_val - 0.1))
        if similar_peaks > 5:  # Too many similar matches
            logger.debug(f"Match rejected: too many similar peaks ({similar_peaks})")
            return False
        
        return True

    def _filter_overlapping_boxes(self, boxes: List[Tuple[int,int,int,int]], scores: List[float]) -> List[Tuple[int,int,int,int]]:
        """Remove overlapping boxes, keeping the one with highest score"""
        if not boxes:
            return []
        
        # Sort by score descending
        sorted_indices = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
        
        filtered_boxes = []
        
        for i in sorted_indices:
            box1 = boxes[i]
            x1, y1, x2, y2 = box1
            
            # Check overlap with already accepted boxes
            overlaps = False
            for accepted_box in filtered_boxes:
                ax1, ay1, ax2, ay2 = accepted_box
                
                # Calculate intersection area
                ix1 = max(x1, ax1)
                iy1 = max(y1, ay1)
                ix2 = min(x2, ax2)
                iy2 = min(y2, ay2)
                
                if ix1 < ix2 and iy1 < iy2:
                    intersection = (ix2 - ix1) * (iy2 - iy1)
                    area1 = (x2 - x1) * (y2 - y1)
                    area2 = (ax2 - ax1) * (ay2 - ay1)
                    
                    # If overlap is more than 30% of either box, consider it overlapping
                    if intersection / min(area1, area2) > 0.3:
                        overlaps = True
                        break
            
            if not overlaps:
                filtered_boxes.append(box1)
        
        return filtered_boxes

    def find(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # 1) preprocess the frame once
        gray = self._preprocess_frame(frame)

        boxes: List[Tuple[int,int,int,int]] = []
        scores: List[float] = []

        # 2) run each patch through matchTemplate with validation
        for idx, tpl in enumerate(self.templates):
            h, w = tpl.shape
            if w > gray.shape[1] or h > gray.shape[0]:
                logger.debug(f"[ContainerFinder] template#{idx} too large for frame, skipping")
                continue

            result = cv2.matchTemplate(gray, tpl, self.method)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            logger.debug(f"[ContainerFinder] template#{idx} ({self.template_names[idx]}) score={max_val:.3f}")

            # Validate the match before accepting
            if self._is_valid_match(result, max_val, max_loc):
                x, y = max_loc
                box = (x, y, x + w, y + h)
                boxes.append(box)
                scores.append(max_val)
                logger.info(f"[ContainerFinder] template#{idx} accepted match at {box} (score={max_val:.3f})")
            else:
                logger.debug(f"[ContainerFinder] template#{idx} match rejected (validation failed)")

        # 3) if none met criteria, bail
        if not boxes:
            logger.warning(f"[ContainerFinder] no valid matches found")
            return None

        # 4) Remove overlapping detections
        filtered_boxes = self._filter_overlapping_boxes(boxes, scores)
        
        if not filtered_boxes:
            logger.warning(f"[ContainerFinder] no matches after overlap filtering")
            return None

        # 5) If only one box, return it. If multiple, union them.
        if len(filtered_boxes) == 1:
            logger.info(f"[ContainerFinder] single container found: {filtered_boxes[0]}")
            return filtered_boxes[0]
        
        # Union multiple boxes
        x1 = min(b[0] for b in filtered_boxes)
        y1 = min(b[1] for b in filtered_boxes)
        x2 = max(b[2] for b in filtered_boxes)
        y2 = max(b[3] for b in filtered_boxes)

        final_box = (x1, y1, x2, y2)
        
        # Sanity check: if union box is too large compared to individual boxes, 
        # return the best scoring box instead
        union_area = (x2 - x1) * (y2 - y1)
        max_individual_area = max((b[2]-b[0])*(b[3]-b[1]) for b in filtered_boxes)
        
        if union_area > max_individual_area * 3:  # Union is 3x larger than biggest component
            best_idx = scores.index(max(score for i, score in enumerate(scores) if boxes[i] in filtered_boxes))
            best_box = boxes[best_idx]
            logger.info(f"[ContainerFinder] union too large, returning best individual match: {best_box}")
            return best_box

        logger.info(f"[ContainerFinder] merged container box = {final_box}")
        return final_box