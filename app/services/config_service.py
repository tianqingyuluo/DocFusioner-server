"""
动态配置服务。

读取链路：schema defaults + DB overrides (JSON) → 内存缓存。
写入链路：DynamicConfigPatch 校验 → 事务写 DB → 刷新缓存。
"""

import asyncio
import json
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import DynamicConfig, DynamicConfigPatch
from app.models.orm import Setting


class DynamicConfigService:
    """
    动态配置管理。

    - load(): 启动时从 DB 加载 overrides，与 schema 默认值合并
    - get(): 业务层调用，直接返回内存缓存
    - update(): PATCH 更新，校验 -> 写 DB -> 刷新缓存
    - reload(): 强制从 DB 重新加载
    """

    def __init__(self):
        self._cache: DynamicConfig | None = None
        self._lock = asyncio.Lock()

    async def load(self, session: AsyncSession) -> DynamicConfig:
        """启动时从 DB 加载 overrides，与 schema 默认值合并。"""
        allowed = set(DynamicConfig.model_fields.keys())
        stmt = select(Setting).where(Setting.key.in_(allowed))
        rows = await session.execute(stmt)

        overrides: dict[str, object] = {}
        for row in rows.scalars():
            try:
                overrides[row.key] = json.loads(row.value)
            except (json.JSONDecodeError, TypeError):
                pass

        self._cache = DynamicConfig(**overrides)
        return self._cache

    def get(self) -> DynamicConfig:
        """业务层调用：直接返回内存缓存。"""
        if self._cache is None:
            return DynamicConfig()
        return self._cache

    async def update(self, session: AsyncSession, patch: DynamicConfigPatch) -> DynamicConfig:
        """PATCH 更新：校验 -> 事务写 DB -> 刷新缓存。"""
        updates = patch.model_dump(exclude_none=True)
        if not updates:
            return self.get()

        async with self._lock:
            current = self.get().model_dump()
            current.update(updates)
            validated = DynamicConfig(**current)

            now = datetime.now(timezone.utc).replace(tzinfo=None)
            for key, value in updates.items():
                await session.merge(
                    Setting(
                        key=key,
                        value=json.dumps(value, ensure_ascii=False),
                        updated_at=now,
                    ),
                )
            await session.commit()

            self._cache = validated
            return validated

    async def reload(self, session: AsyncSession) -> DynamicConfig:
        """强制从 DB 重新加载。"""
        async with self._lock:
            return await self.load(session)


dynamic_config_service = DynamicConfigService()
