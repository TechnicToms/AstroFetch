"""Body-agnostic data access: STAC search, windowed COG reads, gridding, cache.

Moon-specific wiring (which collection backs which product) lives under
``astrofetch/moon``; everything here is body-neutral and reused as new bodies
are added.
"""
