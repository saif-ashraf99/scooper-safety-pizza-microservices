import cv2
import numpy as np
import json
from typing import Optional, List, Dict, Tuple
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContainerFinder:
    def __init__(
        self,
        camera_id: str,
        rois_file: Optional[str] = None,
    ):
        base = os.path.dirname(__file__)
        if rois_file is None:
            rois_file = os.path.join(base, "rois", "container_rois.json")
            
        self.camera_id = camera_id
        self.rois_file = rois_file
        self.static_rois: List[Tuple[int, int, int, int]] = []
        self.load_static_rois()

        logger.info(f"[ContainerFinder] Initialized for camera_id: {self.camera_id}. Static ROIs loaded: {len(self.static_rois) > 0}")

    def load_static_rois(self):
        try:
            with open(self.rois_file, 'r') as f:
                all_rois = json.load(f)
                if self.camera_id in all_rois:
                    self.static_rois = [tuple(roi) for roi in all_rois[self.camera_id]]
                    logger.info(f"[ContainerFinder] Loaded static ROIs for camera {self.camera_id}: {self.static_rois}")
        except FileNotFoundError:
            logger.info(f"[ContainerFinder] ROIs file {self.rois_file} not found. Starting fresh.")
        except json.JSONDecodeError:
            logger.warning(f"[ContainerFinder] Error decoding {self.rois_file}. File might be corrupted.")
        except Exception as e:
            logger.error(f"[ContainerFinder] An unexpected error occurred while loading ROIs: {e}")

    def save_static_rois(self):
        all_rois = {}
        try:
            with open(self.rois_file, 'r') as f:
                all_rois = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"[ContainerFinder] Starting with an empty ROIs file {self.rois_file}.")
        except Exception as e:
            logger.error(f"[ContainerFinder] An unexpected error occurred while loading existing ROIs: {e}")

        all_rois[self.camera_id] = self.static_rois
        with open(self.rois_file, 'w') as f:
            json.dump(all_rois, f, indent=4)
        logger.info(f"[ContainerFinder] Saved static ROIs for camera {self.camera_id}: {self.static_rois}")

    def calibrate(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        logger.info("[ContainerFinder] Starting calibration. Draw bounding boxes around each protein container. Press \'c\' to confirm, \'r\' to reset, \'q\' to quit.")

        clone = frame.copy()
        rois = []
        ix, iy = -1, -1
        drawing = False

        def draw_roi(event, x, y, flags, param):
            nonlocal ix, iy, drawing, clone, rois

            if event == cv2.EVENT_LBUTTONDOWN:
                ix, iy = x, y
                drawing = True

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    img = clone.copy()
                    cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), 2)
                    cv2.imshow('Calibrate Containers', img)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                x1, y1, x2, y2 = min(ix, x), min(iy, y), max(ix, x), max(iy, y)
                rois.append((x1, y1, x2, y2))
                cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('Calibrate Containers', clone)

        cv2.namedWindow('Calibrate Containers')
        cv2.setMouseCallback('Calibrate Containers', draw_roi)

        while True:
            cv2.imshow('Calibrate Containers', clone)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('c'):  # Confirm and save ROIs
                self.static_rois = rois
                self.save_static_rois()
                logger.info(f"[ContainerFinder] Calibration complete. Static ROIs set to: {self.static_rois}")
                break
            elif key == ord('r'):  # Reset ROIs
                rois = []
                clone = frame.copy()
                logger.info("[ContainerFinder] ROIs reset. Please redraw.")
            elif key == ord('q'):  # Quit without saving
                logger.info("[ContainerFinder] Calibration aborted. No ROIs saved.")
                rois = [] # Clear any drawn ROIs if quitting without saving
                break

        cv2.destroyWindow('Calibrate Containers')
        return self.static_rois

    def find(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self.static_rois:
            logger.debug(f"[ContainerFinder] Using static ROIs: {self.static_rois}")
            return self.static_rois
        else:
            logger.warning("[ContainerFinder] No static ROIs found. Calibration may be needed.")
            return []


