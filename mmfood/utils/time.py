from __future__ import annotations

from datetime import datetime, timezone


def to_unix_ts(dt: datetime) -> int:
    """Convert datetime to Unix timestamp (seconds). Defaults to UTC if naive."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())
