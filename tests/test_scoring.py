"""Tests for scoring and result aggregation (paper metrics: full accuracy + text similarity)."""
import csv
import difflib
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def _load_score_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "score_ascii_results",
        os.path.join(os.path.dirname(__file__), "..", "scripts", "score_ascii_results.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_ascii_csv(path: str, rows: list[tuple[str, str, float]]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual Solution", "Predicted Solution", "Response Time (s)"])
        for actual, predicted, t in rows:
            writer.writerow([actual, predicted, f"{t:.4f}"])


def test_text_similarity_full_match():
    mod = _load_score_mod()
    assert mod.text_similarity("abc", "abc") == pytest.approx(1.0)


def test_text_similarity_no_match():
    mod = _load_score_mod()
    assert mod.text_similarity("abc", "xyz") == pytest.approx(0.0)


def test_text_similarity_partial():
    mod = _load_score_mod()
    sim = mod.text_similarity("abcdef", "abcxxx")
    assert 0.0 < sim < 1.0


def test_text_similarity_matches_difflib():
    """text_similarity must match difflib.SequenceMatcher (paper's stated metric)."""
    mod = _load_score_mod()
    for a, b in [("hello", "hxllo"), ("abc123", "abc456"), ("captcha", "captcxa")]:
        expected = difflib.SequenceMatcher(None, a, b).ratio()
        assert mod.text_similarity(a, b) == pytest.approx(expected)


def test_score_file_exact_match():
    mod = _load_score_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model-results.csv")
        _write_ascii_csv(path, [
            ("abc123", "abc123", 1.0),
            ("xyz789", "xyz789", 1.2),
            ("hello1", "wrong1", 0.9),
        ])
        stats = mod.score_file(path)
        assert stats["count"] == 3
        assert stats["full_accuracy"] == pytest.approx(2 / 3, rel=1e-4)


def test_score_file_similarity_computed():
    mod = _load_score_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model-results.csv")
        _write_ascii_csv(path, [
            ("abc", "abc", 1.0),  # similarity = 1.0
            ("abc", "xyz", 1.0),  # similarity = 0.0
        ])
        stats = mod.score_file(path)
        assert stats["text_similarity"] == pytest.approx(0.5)


def test_score_file_all_wrong():
    mod = _load_score_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model-results.csv")
        _write_ascii_csv(path, [
            ("abc123", "xxxxxx", 1.0),
            ("xyz789", "yyyyyy", 1.2),
        ])
        stats = mod.score_file(path)
        assert stats["full_accuracy"] == 0.0


def test_score_file_all_correct():
    mod = _load_score_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model-results.csv")
        _write_ascii_csv(path, [("abc123", "abc123", 1.0)])
        stats = mod.score_file(path)
        assert stats["full_accuracy"] == 1.0
        assert stats["text_similarity"] == pytest.approx(1.0)


def test_score_file_avg_response_time():
    mod = _load_score_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model-results.csv")
        _write_ascii_csv(path, [
            ("abc", "abc", 2.0),
            ("xyz", "xyz", 4.0),
        ])
        stats = mod.score_file(path)
        assert stats["avg_response_time"] == pytest.approx(3.0)


def test_score_empty_file():
    mod = _load_score_mod()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model-results.csv")
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["Actual Solution", "Predicted Solution", "Response Time (s)"])
        stats = mod.score_file(path)
        assert stats["count"] == 0
        assert stats["full_accuracy"] == 0.0


def test_mock_eval_end_to_end():
    """End-to-end: generate samples, run mock eval, score results."""
    import importlib.util

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    base = os.path.join(os.path.dirname(__file__), "..")
    gen_mod = _load("gen", os.path.join(base, "scripts", "generate_ascii_samples.py"))
    mock_mod = _load("mock", os.path.join(base, "scripts", "mock_eval.py"))
    score_mod = _load("score", os.path.join(base, "scripts", "score_ascii_results.py"))

    with tempfile.TemporaryDirectory() as tmpdir:
        import random
        random.seed(0)

        data_dir = os.path.join(tmpdir, "samples")
        os.makedirs(data_dir)
        for _ in range(5):
            label = gen_mod.random_label(5)
            art = gen_mod.render_ascii(label)
            with open(os.path.join(data_dir, f"{label}.txt"), "w") as f:
                f.write(art)

        out_dir = os.path.join(tmpdir, "results")
        mock_mod.run_mock_eval(data_dir, out_dir, ["mock/model-a"], seed=42)

        raw_dir = os.path.join(out_dir, "raw")
        csv_files = [f for f in os.listdir(raw_dir) if f.endswith(".csv")]
        assert len(csv_files) == 1

        stats = score_mod.score_file(os.path.join(raw_dir, csv_files[0]))
        assert stats["count"] >= 0
        assert 0.0 <= stats["full_accuracy"] <= 1.0
        assert 0.0 <= stats["text_similarity"] <= 1.0
