"""Raw BGR frame -> model-ready tensor.

Single source of truth for image preprocessing. Runs identically in the
simulator and (later) on the Jetson, so keep this Python 3.6-compatible.

Steps: crop off sky (top) and bumper/chassis (bottom) -> resize -> normalize to
[-1, 1] -> reorder HWC to CHW. Channel order stays BGR throughout (OpenCV is
BGR-native and CARLA's BGRA drops its alpha to BGR); never swap to RGB on only
one side.
"""
import cv2
import numpy as np

# Defaults assume a 640x360 BGR input (the sim camera and the car's GStreamer
# output are both configured to that). Crops are in pixel rows; recalibrate
# CROP_TOP against real footage during the future hardware plan.
CROP_TOP = 130
CROP_BOTTOM = 30
OUT_SIZE = (200, 66)  # (width, height) fed to the CNN


def preprocess(img_bgr, crop_top=CROP_TOP, crop_bottom=CROP_BOTTOM, out_size=OUT_SIZE):
    """Crop, resize, normalize and reorder a BGR frame to CHW float32 in [-1, 1].

    Args:
        img_bgr: ``(H, W, 3)`` uint8 array in BGR order.
        crop_top: pixel rows removed from the top (sky/horizon).
        crop_bottom: pixel rows removed from the bottom (bumper/chassis).
        out_size: ``(width, height)`` the CNN expects.

    Returns:
        ``np.ndarray`` of shape ``(3, out_height, out_width)``, dtype float32,
        in [-1, 1], channel order BGR.
    """
    h = img_bgr.shape[0]
    cropped = img_bgr[crop_top:h - crop_bottom, :, :]
    resized = cv2.resize(cropped, out_size, interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 127.5 - 1.0
    return np.transpose(norm, (2, 0, 1))
