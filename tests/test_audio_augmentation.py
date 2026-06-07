"""Tests for audio augmentation functions in src/audio-captcha/util.py."""
import importlib.util as _ilu
import io
import os
import sys
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile


def _load_audio_util():
    """Load src/audio-captcha/util.py with a unique module name to avoid sys.modules collision."""
    src_path = os.path.join(os.path.dirname(__file__), "..", "src", "audio-captcha", "util.py")
    spec = _ilu.spec_from_file_location("audio_captcha_util", src_path)
    mod = _ilu.module_from_spec(spec)
    # Ensure the audio-captcha directory is findable for its local imports (log.py)
    audio_src = os.path.dirname(src_path)
    if audio_src not in sys.path:
        sys.path.insert(0, audio_src)
    spec.loader.exec_module(mod)
    return mod


def _make_wav(duration_s: float = 1.0, sample_rate: int = 24000, freq: float = 440.0) -> str:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    samples = (np.sin(2 * np.pi * freq * t) * 16000).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wavfile.write(tmp.name, sample_rate, samples)
    tmp.close()
    return tmp.name


def test_add_gaussian_noise_returns_bytes():
    util = _load_audio_util()
    wav_path = _make_wav()
    try:
        result = util.add_gaussian_noise(wav_path, noise_level=0.1)
        assert isinstance(result, bytes)
        assert len(result) > 0
    finally:
        os.unlink(wav_path)


def test_add_gaussian_noise_preserves_sample_rate():
    util = _load_audio_util()
    wav_path = _make_wav(sample_rate=24000)
    try:
        result_bytes = util.add_gaussian_noise(wav_path, noise_level=0.1)
        rate, _ = wavfile.read(io.BytesIO(result_bytes))
        assert rate == 24000
    finally:
        os.unlink(wav_path)


def test_add_gaussian_noise_changes_samples():
    util = _load_audio_util()
    wav_path = _make_wav()
    try:
        _, clean = wavfile.read(wav_path)
        noisy_bytes = util.add_gaussian_noise(wav_path, noise_level=1.0)
        _, noisy = wavfile.read(io.BytesIO(noisy_bytes))
        assert not np.array_equal(clean, noisy)
    finally:
        os.unlink(wav_path)


def test_add_background_noise_returns_bytes():
    util = _load_audio_util()
    target = _make_wav(duration_s=1.0, freq=440.0)
    background = _make_wav(duration_s=2.0, freq=220.0)
    try:
        result = util.add_background_noise(target, background, boost=1.0)
        assert isinstance(result, bytes)
        assert len(result) > 0
    finally:
        os.unlink(target)
        os.unlink(background)


def test_add_background_noise_sample_rate_mismatch_raises():
    util = _load_audio_util()
    target = _make_wav(sample_rate=24000)
    background = _make_wav(sample_rate=16000)
    try:
        with pytest.raises(ValueError, match="Sample rates"):
            util.add_background_noise(target, background, boost=1.0)
    finally:
        os.unlink(target)
        os.unlink(background)


def test_combine_audio_files_returns_bytes():
    util = _load_audio_util()
    base = _make_wav(duration_s=1.0, freq=440.0)
    other1 = _make_wav(duration_s=1.0, freq=550.0)
    other2 = _make_wav(duration_s=1.0, freq=660.0)
    try:
        result = util.combine_audio_files(base, [other1, other2], ratios=[0.5, 0.5])
        assert isinstance(result, bytes)
        assert len(result) > 0
    finally:
        os.unlink(base)
        os.unlink(other1)
        os.unlink(other2)


def test_audio_to_base64_bytes():
    import base64
    util = _load_audio_util()
    sample_bytes = b"\x00\x01\x02\x03" * 100
    result = util.audio_to_base64_bytes(sample_bytes)
    assert base64.b64decode(result) == sample_bytes
