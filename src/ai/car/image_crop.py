"""Centre-crop the car's camera frame to approximate the training field of view.

The car's lens is 130 deg; the model trained on 62.2 deg. Cropping the centre
brings the framing closer. The right fraction depends on the lens projection
(~0.28 if rectilinear, ~0.48 if closer to equidistant/fisheye) and has NOT been
measured yet -- hence a parameter, with 1.0 meaning "no crop".

Cropping happens BEFORE the resize to the model size, so as little resolution as
possible is thrown away. Python 3.6-safe.
"""
import cv2


def center_crop(frame_bgr, frac):
    """Keep the central ``frac`` of width AND height (aspect ratio preserved).

    The same fraction on both axes matters: stretching one axis would change the
    scene geometry, and the network never saw a stretched world.
    """
    if frac <= 0.0:
        raise ValueError("crop fraction must be > 0, got {0!r}".format(frac))
    if frac >= 1.0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    return frame_bgr[y0:y0 + ch, x0:x0 + cw]


def prepare_frame(frame_bgr, crop_frac, out_size=(640, 360)):
    """Crop the centre then resize to what the model pipeline expects.

    ``out_size`` is ``(width, height)`` -- 640x360 is what ``ai.shared.image_pipeline``
    assumes, and its 130/30 row crops are calibrated against that.
    """
    cropped = center_crop(frame_bgr, crop_frac)
    return cv2.resize(cropped, out_size, interpolation=cv2.INTER_AREA)
