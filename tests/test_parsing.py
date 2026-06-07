"""Tests for response parsing and validation (ASCII CAPTCHA).

validate_output is inlined here so tests run without importing util.py
and triggering its top-level openai dependency.
"""
import re

import pytest


def validate_output(text: str):
    """Mirror of ASCIICaptchaTester.validate_output in src/ascii-captcha/util.py."""
    text = text.strip()
    if not text or len(text) == 0:
        return None
    elif len(text) > 25:
        return None
    elif re.search(r"[.!?]", text):
        return None
    elif re.match(r"\[ERROR", text):
        return None
    elif "\n" in text:
        return None
    elif "#" in text:
        return None
    else:
        return text


def test_clean_short_string_passes():
    assert validate_output("AB12CD") == "AB12CD"


def test_empty_string_returns_none():
    assert validate_output("") is None
    assert validate_output("   ") is None


def test_too_long_returns_none():
    assert validate_output("A" * 26) is None


def test_sentence_with_period_returns_none():
    assert validate_output("The answer is ABC123.") is None


def test_exclamation_mark_returns_none():
    assert validate_output("ABC!") is None


def test_question_mark_returns_none():
    assert validate_output("ABC?") is None


def test_error_prefix_returns_none():
    assert validate_output("[ERROR: something went wrong]") is None


def test_newline_returns_none():
    assert validate_output("AB\nCD") is None


def test_hash_returns_none():
    assert validate_output("AB#CD") is None


def test_strips_whitespace():
    assert validate_output("  AB12  ") == "AB12"


def test_exactly_25_chars_passes():
    assert validate_output("A" * 25) == "A" * 25


def test_exactly_26_chars_fails():
    assert validate_output("A" * 26) is None
