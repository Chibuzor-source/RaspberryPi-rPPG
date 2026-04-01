"""Detect the lower face with MediaPipe's FaceMesh detector.

This detector is based on the [face landmarker task from
MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python).
The face landmarker provides locations of more than 450 facial landmarks.
From these, we can define a region for the lower face, as is done for example
by Li et al. (2014)[^1].

[^1]: X. Li, J. Chen, G. Zhao, and M. Pietikainen, "Remote Heart Rate
    Measurement From Face Videos Under Realistic Situations", Proceedings of
    the IEEE Conference on Computer Vision and Pattern Recognition (CVPR),
    pp. 4264-4271, 2014
    [doi:10.1109/CVPR.2014.543](https://doi.org/10.1109/CVPR.2014.543)
"""

import time
import warnings

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python.components.containers import (
    landmark as landmark_module,  # type: ignore
)

from ..containers import RegionOfInterest
from ..helpers import get_cached_resource_path
from .detector import RoiDetector
from .roi_tools import contour_to_mask

MEDIAPIPE_MODELS_BASE = "https://storage.googleapis.com/mediapipe-models/"
LANDMARKER_TASK = "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

TESSELATION_SPEC = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()  # type: ignore
CONTOUR_SPEC = mp.solutions.drawing_styles.get_default_face_mesh_contours_style()  # type: ignore
IRISES_SPEC = mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style()  # type: ignore


def get_face_landmarker_modelfile():
    """Get the filename of the FaceLandmarker - download file if necessary."""
    task_filename = "face_landmarker.task"
    return get_cached_resource_path(
        task_filename, MEDIAPIPE_MODELS_BASE + LANDMARKER_TASK
    )


def get_landmark_coords(
    landmarks: list[landmark_module.NormalizedLandmark], width: int, height: int
) -> np.ndarray:
    """Extract normalized landmark coordinates to array of pixel coordinates."""
    xyz = [(lm.x, lm.y, lm.z) for lm in landmarks]
    return np.multiply(xyz, [width, height, width]).astype(int)


def get_boundingbox_from_coords(coords: np.ndarray) -> np.ndarray:
    """Calculate the bounding rectangle containing all landmarks."""
    xy = np.min(coords, axis=0)
    wh = np.subtract(np.max(coords, axis=0), xy)
    return np.r_[xy, wh]


# ---------------------------------------------------------------------------
# ROI landmark index definitions
# Each list is a polygon of Face Mesh landmark indices (out of 468 total)
# that outlines a distinct facial region.  You can verify / adjust these
# visually using the MediaPipe Face Mesh landmark map:
# https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png
# ---------------------------------------------------------------------------
ROI_LANDMARK_SETS: dict[str, list[int]] = {
    # Original lower-face polygon — preserved exactly so existing pipeline is unchanged
    "lower_face": [200, 431, 411, 340, 349, 120, 111, 187, 211],
    # Forehead band between hairline and brow line
    "forehead": [10, 109, 67, 103, 54, 21, 162, 127, 234,
                 93, 132, 58, 172, 136, 150, 149, 176, 148, 152],
    # Left cheek (subject's left — appears on RIGHT side of the image)
    "left_cheek": [117, 118, 119, 100, 126, 209, 49, 131, 134, 51, 5, 4, 1, 19, 94],
    # Right cheek (subject's right — appears on LEFT side of the image)
    "right_cheek": [346, 347, 348, 329, 355, 429, 279, 360, 363, 281, 5, 4, 1, 19, 94],
    # Nose bridge and tip
    "nose": [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164, 0, 11, 12, 13, 14, 15, 16, 17, 18],
}

# BGR colours for bounding-box overlays, one per ROI name
ROI_COLORS: dict[str, tuple[int, int, int]] = {
    "lower_face":  (0, 255, 0),    # green
    "forehead":    (255, 100, 0),  # blue
    "left_cheek":  (0, 165, 255),  # orange
    "right_cheek": (0, 0, 255),    # red
    "nose":        (255, 255, 0),  # cyan
}


class FaceMeshDetector(RoiDetector):
    """Face detector using MediaPipe's face landmarker.

    Extends the original single-ROI detector to support multiple named ROIs
    (lower_face, forehead, left_cheek, right_cheek, nose).

    Backward compatibility:
    - ``RegionOfInterest.mask`` still holds the lower-face polygon mask, so
      the existing ``Processor`` / ``HrCalculator`` chain works without changes.
    - Per-ROI masks are attached as ``RegionOfInterest.roi_masks`` (a dict)
      for optional downstream use (next milestone: parallel signal extraction).

    New constructor args:
        draw_roi_boxes (bool): if True, draw coloured labelled bounding boxes
            for every ROI on the live camera frame. Defaults to True.
    """

    _lower_face = ROI_LANDMARK_SETS["lower_face"]  # kept for back-compat

    def __init__(self, draw_landmarks: bool = False, draw_roi_boxes: bool = True, **kwargs):
        super().__init__(**kwargs)
        modelpath = get_face_landmarker_modelfile()
        if modelpath is None:
            raise FileNotFoundError("Could not find or download landmarker model file.")
        base_options = mp.tasks.BaseOptions(model_asset_path=modelpath)
        landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            landmarker_options
        )
        self.draw_landmarks = draw_landmarks
        self.draw_roi_boxes = draw_roi_boxes

    def __del__(self):
        self.landmarker.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_roi_masks(
        self, coords: np.ndarray, height: int, width: int
    ) -> dict[str, np.ndarray]:
        """Build a binary mask (0/1 uint8) for every entry in ROI_LANDMARK_SETS.

        Args:
            coords: (N, 2) array of pixel-space landmark coordinates.
            height: frame height in pixels.
            width:  frame width in pixels.

        Returns:
            Dict mapping ROI name → binary mask array.
        """
        return {
            name: contour_to_mask((height, width), coords[indices])
            for name, indices in ROI_LANDMARK_SETS.items()
        }

    def _draw_roi_boxes(
        self, frame: np.ndarray, masks: dict[str, np.ndarray]
    ) -> None:
        """Draw coloured labelled bounding boxes on *frame* in-place.

        For each ROI mask the tightest axis-aligned bounding rectangle is
        computed and drawn with its label just above the top-left corner.

        Args:
            frame: BGR image to annotate (modified in-place).
            masks: dict of ROI name → binary (0/1) mask.
        """
        for name, mask in masks.items():
            color = ROI_COLORS.get(name, (255, 255, 255))
            # findContours needs a 0/255 uint8 image
            mask_u8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            x, y, w, h = cv2.boundingRect(contours[0])
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            # Clamp label y so it never goes off the top of the frame
            label_y = max(y - 6, 12)
            cv2.putText(
                frame,
                name,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    def _process_landmarks(
        self, frame: np.ndarray, results
    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Compute pixel coords, primary mask, face rect, and all ROI masks.

        Returns:
            mask:      lower-face binary mask (same semantics as original).
            face_rect: [x, y, w, h] bounding box of the whole face.
            roi_masks: dict of all named ROI masks.
        """
        height, width = frame.shape[:2]
        coords = get_landmark_coords(results.face_landmarks[0], width, height)[:, :2]
        face_rect = get_boundingbox_from_coords(coords)

        # Primary mask — unchanged from original implementation
        mask = contour_to_mask((height, width), coords[self._lower_face])

        # Build every ROI mask in one pass
        roi_masks = self._build_roi_masks(coords, height, width)

        return mask, face_rect, roi_masks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> RegionOfInterest:
        """Find face landmarks, build all ROI masks, and optionally draw boxes.

        Returns a ``RegionOfInterest`` that is fully backward-compatible with
        the rest of the yarppg pipeline.  The extra ``roi_masks`` attribute
        (dict) carries per-ROI masks for the next milestone (parallel signal
        extraction in ``rppg.py``).
        """
        rawimg = frame.copy()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = self.landmarker.detect_for_video(
                mp_image, int(time.perf_counter() * 1000)
            )

        if len(results.face_landmarks) < 1:
            empty_roi = RegionOfInterest(np.zeros_like(frame), baseimg=frame)
            empty_roi.roi_masks = {}  # type: ignore[attr-defined]
            return empty_roi

        if self.draw_landmarks:
            self.draw_facemesh(frame, results.face_landmarks[0], tesselate=True)

        mask, face_rect, roi_masks = self._process_landmarks(frame, results)

        if self.draw_roi_boxes:
            self._draw_roi_boxes(frame, roi_masks)

        detected_roi = RegionOfInterest(
            mask, baseimg=rawimg, face_rect=tuple(face_rect)
        )
        # Attach multi-ROI masks for downstream use (rppg.py next milestone)
        detected_roi.roi_masks = roi_masks  # type: ignore[attr-defined]
        return detected_roi

    def draw_facemesh(
        self,
        img,
        face_landmarks,
        tesselate=False,
        contour=False,
        irises=False,
    ):
        """Draw the detected face landmarks on the image."""
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()  # type: ignore
        face_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(  # type: ignore
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in face_landmarks
            ]
        )
        if tesselate:
            mp.solutions.drawing_utils.draw_landmarks(  # type: ignore
                image=img,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,  # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=TESSELATION_SPEC,
            )
        if contour:
            mp.solutions.drawing_utils.draw_landmarks(  # type: ignore
                image=img,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,  # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=CONTOUR_SPEC,
            )
        if irises:
            mp.solutions.drawing_utils.draw_landmarks(  # type: ignore
                image=img,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,  # type: ignore
                landmark_drawing_spec=None,
                connection_drawing_spec=IRISES_SPEC,
            )

