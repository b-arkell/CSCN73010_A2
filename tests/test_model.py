"""Tests for image preprocessing and prediction in model.py."""
import io
from pathlib import Path

import pytest
from PIL import Image

from model import preprocess_img, predict_result

TEST_IMAGES = Path(__file__).resolve().parent.parent / "test_images"


def _make_png(mode, size=(100, 80)):
    """Create an in-memory PNG with the given colour mode."""
    buf = io.BytesIO()
    Image.new(mode, size).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_preprocess_output_shape_and_range():
    """Normal case: an RGB image becomes a 1x224x224x3 array scaled to 0-1."""
    arr = preprocess_img(_make_png("RGB"))
    assert arr.shape == (1, 224, 224, 3)
    assert arr.min() >= 0.0
    assert arr.max() <= 1.0


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_preprocess_handles_non_rgb_images(mode):
    """Edge case: grayscale (L) and transparent (RGBA) images should still work."""
    arr = preprocess_img(_make_png(mode))
    assert arr.shape == (1, 224, 224, 3)


def test_preprocess_rejects_non_image():
    """Edge case: bytes that aren't an image raise an OSError."""
    with pytest.raises(OSError):
        preprocess_img(io.BytesIO(b"this is not an image"))


def test_real_model_predicts_a_digit():
    """Integration: a real sample image gives a digit between 0 and 9."""
    image_types = {".jpg", ".jpeg", ".png"}
    sample = next(p for p in TEST_IMAGES.rglob("*") if p.suffix.lower() in image_types)
    prediction = predict_result(preprocess_img(sample))
    assert 0 <= int(prediction) <= 9
