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
    ):
        # 1) locate the folder of patches
        base = os.path.dirname(__file__)
        imgs_folder = tpl_dir or os.path.join(base, "imgs")
        if not os.path.isdir(imgs_folder):
            raise FileNotFoundError(f"[ContainerFinder] no imgs folder at {imgs_folder}")

        # 2) load every .png in there as a separate template
        self.templates: List[np.ndarray] = []
        for fn in sorted(os.listdir(imgs_folder)):
            if fn.lower().endswith(".png"):
                path = os.path.join(imgs_folder, fn)
                tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if tpl is None:
                    raise IOError(f"[ContainerFinder] failed to load {path}")
                self.templates.append(tpl)
                logger.info(f"[ContainerFinder] loaded patch {fn} size={tpl.shape[::-1]}")
        if not self.templates:
            raise ValueError(f"[ContainerFinder] no .png templates found in {imgs_folder}")

        self.th = match_threshold
        self.use_edges = use_edges
        self.blur_ksize = blur_ksize

        # Precompute edge versions if requested
        if self.use_edges:
            self.templates = [cv2.Canny(tpl, 50, 150) for tpl in self.templates]

    def find(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        # 1) preprocess the frame once
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, self.blur_ksize, 0)
        if self.use_edges:
            gray = cv2.Canny(gray, 50, 150)

        boxes: List[Tuple[int,int,int,int]] = []

        # 2) run each patch through matchTemplate
        for idx, tpl in enumerate(self.templates):
            h, w = tpl.shape
            if w > gray.shape[1] or h > gray.shape[0]:
                logger.debug(f"[ContainerFinder] patch#{idx} too large for frame, skipping")
                continue

            res = cv2.matchTemplate(gray, tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            logger.debug(f"[ContainerFinder] patch#{idx} score={max_val:.3f}")

            if max_val >= self.th:
                x, y = max_loc
                box = (x, y, x + w, y + h)
                boxes.append(box)
                logger.info(f"[ContainerFinder] patch#{idx} match at {box} (score={max_val:.3f})")

        # 3) if none met threshold, bail
        if not boxes:
            logger.warning(f"[ContainerFinder] no patches passed threshold {self.th}")
            return None

        # 4) union all boxes into one bounding rectangle
        x1 = min(b[0] for b in boxes)
        y1 = min(b[1] for b in boxes)
        x2 = max(b[2] for b in boxes)
        y2 = max(b[3] for b in boxes)

        logger.info(f"[ContainerFinder] merged container box = ({x1},{y1},{x2},{y2})")
        return (x1, y1, x2, y2)
