"""Live endpoint tests — hit real government servers, manual trigger only.

These never run in CI or in the default ``pytest`` invocation; they are
deselected by the ``-m 'not live'`` default in ``pyproject.toml``. Run them
explicitly, one at a time, when verifying a real endpoint::

    uv run pytest tests/live -m live

Phase 0 ships no network code yet, so this module only establishes the policy
and the ``live`` marker. Real STAC/COG connectivity checks land with the
Phase 1 sampler.
"""

import pytest


@pytest.mark.live
def test_live_marker_is_registered() -> None:
    """Placeholder asserting the live suite is wired up.

    Replace with a real USGS ARD STAC reachability check in Phase 1.
    """
    pytest.skip("no live endpoints until the Phase 1 STAC sampler lands")
