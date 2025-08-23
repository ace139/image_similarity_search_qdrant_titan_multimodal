from __future__ import annotations

import hashlib


def md5_hex(b: bytes) -> str:
    """Return hex MD5 digest of bytes."""
    m = hashlib.md5()
    m.update(b)
    return m.hexdigest()
