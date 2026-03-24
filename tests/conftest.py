import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_910b: requires 910B hardware")


def _check_910b_available():
    """Return True if a 910B-family device is accessible."""
    try:
        from candle_dvm.system import System
        sys = System()
        sys.init(0)
        return sys.arch_name() == "c220"
    except Exception:
        return False


# Cache the result so we only probe once per session.
_has_910b = None


def pytest_collection_modifyitems(config, items):
    """Auto-skip tests marked ``requires_910b`` when no 910B device is present."""
    global _has_910b
    needs_check = any("requires_910b" in item.keywords for item in items)
    if not needs_check:
        return

    if _has_910b is None:
        _has_910b = _check_910b_available()

    if _has_910b:
        return

    skip_marker = pytest.mark.skip(reason="requires 910B hardware")
    for item in items:
        if "requires_910b" in item.keywords:
            item.add_marker(skip_marker)
