"""统一时间戳工具"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """返回 UTC naive datetime，所有应用层写入时间统一调用。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)
