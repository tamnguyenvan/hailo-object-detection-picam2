from typing import List, Generator, Optional, Tuple, Dict, Callable ,Any
from pathlib import Path
from loguru import logger
import json
import os
import sys
import numpy as np
import queue
import cv2
import time
import threading
from enum import Enum
from pathlib import Path
import subprocess
import difflib


IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.png', '.bmp', '.jpeg')
CAMERA_INDEX = int(os.environ.get('CAMERA_INDEX', '0'))
RESOURCES_DOWNLOAD_DIR = Path(__file__).resolve().parents[2] / "resources_download"
GET_HEF_BASH_SCRIPT_PATH   = RESOURCES_DOWNLOAD_DIR / "get_hef.sh"
GET_INPUT_BASH_SCRIPT_PATH = RESOURCES_DOWNLOAD_DIR / "get_input.sh"
RESOLUTION_MAP = {
    "sd": (640, 480),
    "hd": (1280, 720),
    "fhd": (1920, 1080)
}

def default_preprocess(image: np.ndarray, model_w: int, model_h: int) -> np.ndarray:
    """
    Resize image with unchanged aspect ratio using padding.

    Args:
        image (np.ndarray): Input image.
        model_w (int): Model input width.
        model_h (int): Model input height.

    Returns:
        np.ndarray: Preprocessed and padded image.
    """
    img_h, img_w, _ = image.shape[:3]
    scale = min(model_w / img_w, model_h / img_h)
    new_img_w, new_img_h = int(img_w * scale), int(img_h * scale)
    image = cv2.resize(image, (new_img_w, new_img_h), interpolation=cv2.INTER_CUBIC)

    padded_image = np.full((model_h, model_w, 3), (114, 114, 114), dtype=np.uint8)
    x_offset = (model_w - new_img_w) // 2
    y_offset = (model_h - new_img_h) // 2
    padded_image[y_offset:y_offset + new_img_h, x_offset:x_offset + new_img_w] = image

    return padded_image



def load_json_file(path: str) -> Dict[str, Any]:
    """
    Loads and parses a JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed contents of the JSON file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        OSError: If the file cannot be read.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Invalid JSON format in file '{path}': {e.msg}", e.doc, e.pos)

    return data

def get_labels(labels_path: str) -> list:
        """
        Load labels from a file.

        Args:
            labels_path (str): Path to the labels file.

        Returns:
            list: List of class names.
        """
        with open(labels_path, 'r', encoding="utf-8") as f:
            class_names = f.read().splitlines()
        return class_names


####################################################################
# Frame Rate Tracker
####################################################################

class FrameRateTracker:
    """
    Tracks frame count and elapsed time to compute real-time FPS (frames per second).
    """

    def __init__(self):
        """Initialize the tracker with zero frames and no start time."""
        self._count = 0
        self._start_time = None

    def start(self) -> None:
        """Start or restart timing and reset the frame count."""
        self._start_time = time.time()

    def increment(self, n: int = 1) -> None:
        """Increment the frame count.

        Args:
            n (int): Number of frames to add. Defaults to 1.
        """
        self._count += n


    @property
    def count(self) -> int:
        """Returns:
            int: Total number of frames processed.
        """
        return self._count

    @property
    def elapsed(self) -> float:
        """Returns:
            float: Elapsed time in seconds since `start()` was called.
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def fps(self) -> float:
        """Returns:
            float: Calculated frames per second (FPS).
        """
        elapsed = self.elapsed
        return self._count / elapsed if elapsed > 0 else 0.0

    def frame_rate_summary(self) -> str:
        """Return a summary of frame count and FPS.

        Returns:
            str: e.g. "Processed 200 frames at 29.81 FPS"
        """
        return f"Processed {self.count} frames at {self.fps:.2f} FPS"