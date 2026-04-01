"""Defines some containers passed between objects of the yarPPG application."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RegionOfInterest:
    """Container for defining the region of interest (and background) in an image."""

    mask: np.ndarray
    baseimg: np.ndarray
    bg_mask: np.ndarray | None = None
    face_rect: tuple[int, int, int, int] | None = None
    """Bounding box of the detected face (x, y, w, h)."""

    roi_masks: dict = field(default_factory=dict)
    """Per-region binary masks keyed by ROI name, e.g. 'forehead', 'left_cheek'.
    Empty dict when the detector does not support multi-ROI or no face was found.
    Added for the multi-ROI milestone; ignored by all existing single-ROI code."""


@dataclass
class Color:
    """Defines a color in RGB(A) format."""

    r: float
    g: float
    b: float

    @classmethod
    def null(cls):
        """Create empty color with NaN values."""
        return cls(np.nan, np.nan, np.nan)

    def __array__(self):
        return np.array([self.r, self.g, self.b])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        """Convert numpy array to `Color` object."""
        if len(arr) in {3, 4} and arr.ndim == 1:
            return cls(*arr)
        raise ValueError(f"Cannot interpret {arr=!r}")


@dataclass
class RppgResult:
    """Container for rPPG computation results.

    Calling `np.array` on this container will return a 8-element vector containing
    the rPPG signal value, RGB values of the ROI, RGB values of the background (or nans)
    and the HR. `to_series` produces a clearer representation of the values with named
    indices.

    Note that both `__array__` and `to_series` ignore the `roi` attribute.
    """

    value: float
    """Output value of the rPPG signal extractor."""

    roi: RegionOfInterest
    """Region of interest identified in the current frame."""

    roi_mean: Color
    """Mean color of the ROI."""

    bg_mean: Color
    """Mean color of the background."""

    hr: float = np.nan
    """Heart rate estimate in frames per beat."""

    roi_signal_means: dict = field(default_factory=dict)
    """Per-ROI mean Color values keyed by ROI name.
    Populated by Rppg.process_frame when multi-ROI masks are available.
    Empty dict for backward-compat when using detectors without multi-ROI support."""

    def __array__(self):
        return np.r_[self.value, self.roi_mean, self.bg_mean, self.hr]

    def to_series(self):
        """Extract the rPPG signal values into a Pandas series."""
        return pd.Series(
            np.array(self),
            index=["value", "roi_r", "roi_g", "roi_b", "bg_r", "bg_g", "bg_b", "hr"],
        )
