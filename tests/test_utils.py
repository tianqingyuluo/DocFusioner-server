"""测试工具函数"""

from datetime import datetime, timezone


def test_utc_now_returns_naive_datetime():
    """utc_now() 应返回无时区信息的 UTC 时间"""
    from app.utils.time_utils import utc_now

    result = utc_now()
    assert isinstance(result, datetime)
    assert result.tzinfo is None


def test_utc_now_is_close_to_real_utc():
    """utc_now() 返回值应接近真实 UTC 时间（误差 < 2 秒）"""
    from app.utils.time_utils import utc_now

    result = utc_now()
    real_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    diff = abs((real_utc - result).total_seconds())
    assert diff < 2.0
