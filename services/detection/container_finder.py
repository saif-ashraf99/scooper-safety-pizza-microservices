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
        tpl_path: Optional[str] = None,
        match_threshold: float = 0.5,
        scales: Optional[List[float]] = None,
        use_edges: bool = False,
        blur_ksize: Tuple[int, int] = (5, 5),
    ):
        # Resolve template path
        if tpl_path:
            # if given path doesn't exist, try relative to this module
            if not os.path.exists(tpl_path):
                candidate = os.path.join(os.path.dirname(__file__), tpl_path)
                if os.path.exists(candidate):
                    tpl_path = candidate
        else:
            base = os.path.dirname(__file__) 
            tpl_path = os.path.join(base, "imgs", "protein_template.png")
            
        # Final exist check
        if not os.path.exists(tpl_path):
            raise FileNotFoundError(f"[ContainerFinder] template not found at {tpl_path}")

        # Load template once (grayscale)
        tpl = cv2.imread(tpl_path, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            raise IOError(f"[ContainerFinder] failed to load template data from {tpl_path}")

        self.tpl_orig = tpl
        self.h0, self.w0 = tpl.shape
        self.th = match_threshold
        self.use_edges = use_edges
        self.blur_ksize = blur_ksize
        self.scales = scales or list(np.linspace(0.5, 1.5, 20))

        if self.use_edges:
            self.tpl_edges = cv2.Canny(self.tpl_orig, 50, 150)

        logger.info(
            f"[ContainerFinder] loaded template {tpl_path} "
            f"size=({self.w0}×{self.h0}), thresh={self.th}, scales={len(self.scales)}"
        )

    def find(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # 1) preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        if self.use_edges:
            gray = cv2.Canny(gray, 50, 150)

        best_score = -1.0
        best_loc = None
        best_w = best_h = 0

        # 2) multi-scale match (resize template, keep frame static)
        for scale in self.scales:
            w = int(self.w0 * scale)
            h = int(self.h0 * scale)
            if w < 1 or h < 1 or w > gray.shape[1] or h > gray.shape[0]:
                continue

            tpl_rs = self.tpl_edges if self.use_edges else self.tpl_orig
            tpl_rs = cv2.resize(tpl_rs, (w, h), interpolation=cv2.INTER_AREA)

            res = cv2.matchTemplate(gray, tpl_rs, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            logger.debug(f"[ContainerFinder] scale={scale:.2f} score={max_val:.3f}")
            if max_val > best_score:
                best_score, best_loc, best_w, best_h = max_val, max_loc, w, h

        # 3) threshold check
        if best_loc is None or best_score < self.th:
            logger.warning(f"[ContainerFinder] no match (best_score={best_score:.3f})")
            return None

        x, y = best_loc
        logger.info(
            f"[ContainerFinder] match @ ({x},{y}) size=({best_w}×{best_h}) "
            f"score={best_score:.3f}"
        )
        return (x, y, x + best_w, y + best_h)
