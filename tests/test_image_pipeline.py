"""Tests for the shared image preprocessing pipeline.

Pins down the contract that turns a raw BGR frame (CARLA drops its BGRA alpha ->
BGR; the car's GStreamer sink is already BGR) into model input: crop, resize,
normalize to [-1, 1], and reorder to CHW. Channel order MUST stay BGR on both
sides (swapping on only one side silently fabricates a sim-to-real gap).
"""
import numpy as np
import pytest

from ai.shared.image_pipeline import preprocess


def _solid(h, w, bgr):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = bgr
    return img


def test_output_shape_and_dtype():
    out = preprocess(_solid(360, 640, (0, 0, 0)))
    assert out.shape == (3, 66, 200)
    assert out.dtype == np.float32


def test_black_maps_to_minus_one():
    out = preprocess(_solid(360, 640, (0, 0, 0)))
    assert np.allclose(out, -1.0)


def test_white_maps_to_plus_one():
    out = preprocess(_solid(360, 640, (255, 255, 255)))
    assert np.allclose(out, 1.0)


def test_preserves_bgr_channel_order():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:, :, 0] = 255  # blue channel in BGR
    out = preprocess(img)
    assert np.allclose(out[0], 1.0)
    assert np.allclose(out[1], -1.0)
    assert np.allclose(out[2], -1.0)


def test_crops_top_rows():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:130, :, :] = 255  # sky band to be cropped away
    out = preprocess(img, crop_top=130, crop_bottom=30)
    assert np.allclose(out, -1.0)


def test_crops_bottom_rows():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[330:, :, :] = 255  # bumper band to be cropped away
    out = preprocess(img, crop_top=130, crop_bottom=30)
    assert np.allclose(out, -1.0)


def test_values_within_range():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    out = preprocess(img)
    assert out.min() >= -1.0
    assert out.max() <= 1.0
