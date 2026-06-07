"""
Pytest configuration.

Injects a minimal openai mock and fake env vars so that util.py modules
can be imported in test environments where:
  - the installed openai package predates AsyncOpenAI (openai < 1.0), and
  - no real API keys are set.

The mock allows module-level client instantiation to succeed but raises
if an actual API call is attempted.
"""
import os
import sys
import types


# Provide placeholder env vars so module-level client instantiation doesn't
# fail with KeyError before tests even run.
_FAKE_ENV_DEFAULTS = {
    "OPENAI_API_KEY": "test-placeholder",
    "OPENROUTER_API_KEY": "test-placeholder",
    "GOOGLE_API_KEY": "test-placeholder",
    "ANTHROPIC_API_KEY": "test-placeholder",
}
for _k, _v in _FAKE_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)


def _ensure_openai_mock():
    """Add AsyncOpenAI shim if the installed openai is too old."""
    try:
        from openai import AsyncOpenAI  # noqa: F401
        return  # modern openai present, nothing to do
    except ImportError:
        pass

    openai_mod = sys.modules.get("openai")
    if openai_mod is None:
        openai_mod = types.ModuleType("openai")
        sys.modules["openai"] = openai_mod

    if not hasattr(openai_mod, "AsyncOpenAI"):
        class _FakeClient:
            """Minimal AsyncOpenAI stub — instantiation succeeds, API calls raise."""
            def __init__(self, *args, **kwargs):
                pass  # allow module-level instantiation

            @property
            def responses(self):
                raise RuntimeError("openai mock: real API calls not allowed in tests")

            @property
            def chat(self):
                raise RuntimeError("openai mock: real API calls not allowed in tests")

        openai_mod.AsyncOpenAI = _FakeClient


_ensure_openai_mock()
