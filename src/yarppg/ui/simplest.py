"""Provides a simplistic user interface with values printed to console only."""
import dataclasses

import cv2

from picamera2 import Picamera2

import yarppg

FONT_COLOR = (207, 117, 6)


def _is_window_closed(name: str) -> bool:
    return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1


@dataclasses.dataclass
class SimplestOpenCvWindowSettings(yarppg.settings.UiSettings):
    """Configuration for the simplest OpenCV user interface."""

    roi_alpha: float = 0.0
    video: int | str = 0


def launch_loop(rppg: yarppg.Rppg, config: SimplestOpenCvWindowSettings) -> int:
    """Launch a simple Qt6-based GUI visualizing rPPG results in real-time."""
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration())
    picam2.start()


    tracker = yarppg.FpsTracker()
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = rppg.process_frame(frame)
        img = yarppg.roi.overlay_mask(
            frame, result.roi.mask != 0, alpha=config.roi_alpha
        )
        img = cv2.flip(img, 1)
        tracker.tick()
        result.hr = 60 * tracker.fps / result.hr
        text = f"{result.hr:.1f} (bpm)"
        pos = (10, img.shape[0] - 10)
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_COMPLEX, 0.8, color=FONT_COLOR)
        cv2.imshow("yarPPG", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(result.value, result.hr)
        if cv2.waitKey(1) == ord("q") or _is_window_closed("yarPPG"):
            break
        
    picam2.stop()
    return 0
