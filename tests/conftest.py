"""Auto-skip ml-marked tests where the real ML stack isn't installed.

Tests marked `ml` trigger real `import torch` calls inside fakemon_forge
(function-local imports in sprites.py), so they can only run where torch is
installed. Environments without it — the keep sandbox container, a CPU-only
CI runner — skip them automatically; no pytest flags needed.
"""

import importlib.util

import pytest

_HAS_TORCH = importlib.util.find_spec("torch") is not None


def pytest_collection_modifyitems(config, items):
    if _HAS_TORCH:
        return
    skip = pytest.mark.skip(
        reason="torch not installed — ml tests run on the host, skips here are expected"
    )
    for item in items:
        if "ml" in item.keywords:
            item.add_marker(skip)
