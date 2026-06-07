"""Tests for ASCII CAPTCHA generation and image rendering."""
import importlib.util as _ilu
import os
import string
import sys
import tempfile

import pytest


def _load_generate_mod():
    spec = _ilu.spec_from_file_location(
        "generate_ascii_samples",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "generate_ascii_samples.py"),
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_ascii_util():
    """Load src/ascii-captcha/util.py (requires conftest openai mock)."""
    ascii_src = os.path.join(os.path.dirname(__file__), "..", "src", "ascii-captcha")
    if ascii_src not in sys.path:
        sys.path.insert(0, ascii_src)
    utils_dir = os.path.join(ascii_src, "utils")
    if utils_dir not in sys.path:
        sys.path.insert(0, utils_dir)

    spec = _ilu.spec_from_file_location(
        "ascii_captcha_util",
        os.path.join(ascii_src, "util.py"),
    )
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_ascii_samples_script_imports():
    mod = _load_generate_mod()
    assert callable(mod.random_label)
    assert callable(mod.render_ascii)


def test_random_label_length():
    mod = _load_generate_mod()
    for length in [4, 6, 8]:
        label = mod.random_label(length)
        assert len(label) == length
        assert all(c in (string.ascii_uppercase + string.digits) for c in label)


def test_render_ascii_nonempty():
    mod = _load_generate_mod()
    result = mod.render_ascii("AB12", font="standard")
    assert isinstance(result, str)
    assert len(result.strip()) > 0


def test_generate_samples_writes_files():
    import random as _random
    mod = _load_generate_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        _random.seed(0)
        for _ in range(5):
            label = mod.random_label(5)
            art = mod.render_ascii(label)
            path = os.path.join(tmpdir, f"{label}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(art)

        files = os.listdir(tmpdir)
        assert len(files) == 5
        for fname in files:
            stem = os.path.splitext(fname)[0]
            assert len(stem) == 5
            assert all(c in (string.ascii_uppercase + string.digits) for c in stem)
            with open(os.path.join(tmpdir, fname), encoding="utf-8") as f:
                content = f.read()
            assert len(content.strip()) > 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "fonts", "courier.ttf")),
    reason="courier.ttf not present",
)
def test_text_to_image_returns_pil_image():
    from PIL import Image
    util = _load_ascii_util()
    sample_art = "  _   _\n | | | |\n |_| |_|\n"
    img = util.text_to_image(sample_art)
    assert isinstance(img, Image.Image)
    assert img.width > 0
    assert img.height > 0


@pytest.mark.skipif(
    not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "fonts", "courier.ttf")),
    reason="courier.ttf not present",
)
def test_text_to_image_produces_bytes():
    import base64
    from io import BytesIO
    util = _load_ascii_util()
    img = util.text_to_image("ABC\nDEF")
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    assert len(b64) > 0
