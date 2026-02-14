"""测试 DynamicConfigService"""

import json

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.orm import Setting


class TestDynamicConfigServiceLoad:
    """load() 测试"""

    @pytest.mark.asyncio
    async def test_load_empty_db_returns_defaults(self, db_session: AsyncSession):
        """DB 无 overrides 时返回 schema 默认值"""
        from app.services.config_service import DynamicConfigService

        svc = DynamicConfigService()
        cfg = await svc.load(db_session)

        assert cfg.llm_provider == "deepseek"
        assert cfg.chunk_size == 800

    @pytest.mark.asyncio
    async def test_load_merges_db_overrides(self, db_session: AsyncSession):
        """DB 中的值应覆盖 schema 默认值"""
        from app.services.config_service import DynamicConfigService

        db_session.add(Setting(key="llm_provider", value='"ollama"'))
        db_session.add(Setting(key="chunk_size", value="1200"))
        await db_session.commit()

        svc = DynamicConfigService()
        cfg = await svc.load(db_session)

        assert cfg.llm_provider == "ollama"
        assert cfg.chunk_size == 1200
        assert cfg.llm_model == "deepseek-chat"

    @pytest.mark.asyncio
    async def test_load_ignores_unknown_keys(self, db_session: AsyncSession):
        """DB 中的非白名单 key 应被忽略"""
        from app.services.config_service import DynamicConfigService

        db_session.add(Setting(key="unknown_key", value='"whatever"'))
        await db_session.commit()

        svc = DynamicConfigService()
        cfg = await svc.load(db_session)
        assert not hasattr(cfg, "unknown_key")

    @pytest.mark.asyncio
    async def test_load_skips_corrupted_json(self, db_session: AsyncSession):
        """损坏的 JSON 值应被跳过，回退到默认值"""
        from app.services.config_service import DynamicConfigService

        db_session.add(Setting(key="chunk_size", value="not_valid_json{"))
        await db_session.commit()

        svc = DynamicConfigService()
        cfg = await svc.load(db_session)
        assert cfg.chunk_size == 800


class TestDynamicConfigServiceGet:
    """get() 测试"""

    def test_get_before_load_returns_defaults(self):
        """未 load 时 get() 返回 schema 默认值"""
        from app.services.config_service import DynamicConfigService

        svc = DynamicConfigService()
        cfg = svc.get()
        assert cfg.llm_provider == "deepseek"

    @pytest.mark.asyncio
    async def test_get_after_load_returns_cached(self, db_session: AsyncSession):
        """load 后 get() 返回缓存"""
        from app.services.config_service import DynamicConfigService

        db_session.add(Setting(key="chunk_size", value="1500"))
        await db_session.commit()

        svc = DynamicConfigService()
        await svc.load(db_session)

        cfg = svc.get()
        assert cfg.chunk_size == 1500


class TestDynamicConfigServiceUpdate:
    """update() 测试"""

    @pytest.mark.asyncio
    async def test_update_single_field(self, db_session: AsyncSession):
        """更新单个字段"""
        from app.config import DynamicConfigPatch
        from app.services.config_service import DynamicConfigService

        svc = DynamicConfigService()
        await svc.load(db_session)

        patch = DynamicConfigPatch(chunk_size=2000)
        cfg = await svc.update(db_session, patch)

        assert cfg.chunk_size == 2000
        assert cfg.llm_provider == "deepseek"

    @pytest.mark.asyncio
    async def test_update_persists_to_db(self, db_session: AsyncSession):
        """更新后值应持久化到 DB"""
        from sqlalchemy import select

        from app.config import DynamicConfigPatch
        from app.services.config_service import DynamicConfigService

        svc = DynamicConfigService()
        await svc.load(db_session)

        patch = DynamicConfigPatch(llm_provider="ollama")
        await svc.update(db_session, patch)

        result = await db_session.execute(
            select(Setting).where(Setting.key == "llm_provider"),
        )
        row = result.scalar_one()
        assert json.loads(row.value) == "ollama"

    @pytest.mark.asyncio
    async def test_update_empty_patch_noop(self, db_session: AsyncSession):
        """空 patch 不做任何操作"""
        from app.config import DynamicConfigPatch
        from app.services.config_service import DynamicConfigService

        svc = DynamicConfigService()
        await svc.load(db_session)

        original = svc.get().model_dump()
        patch = DynamicConfigPatch()
        cfg = await svc.update(db_session, patch)

        assert cfg.model_dump() == original

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, db_session: AsyncSession):
        """同时更新多个字段"""
        from app.config import DynamicConfigPatch
        from app.services.config_service import DynamicConfigService

        svc = DynamicConfigService()
        await svc.load(db_session)

        patch = DynamicConfigPatch(
            llm_provider="ollama",
            chunk_size=1500,
            chunk_overlap=200,
        )
        cfg = await svc.update(db_session, patch)

        assert cfg.llm_provider == "ollama"
        assert cfg.chunk_size == 1500
        assert cfg.chunk_overlap == 200


class TestDynamicConfigServiceReload:
    """reload() 测试"""

    @pytest.mark.asyncio
    async def test_reload_refreshes_cache(self, db_session: AsyncSession):
        """reload 应重新从 DB 加载"""
        from app.services.config_service import DynamicConfigService

        svc = DynamicConfigService()
        await svc.load(db_session)
        assert svc.get().chunk_size == 800

        db_session.add(Setting(key="chunk_size", value="999"))
        await db_session.commit()

        cfg = await svc.reload(db_session)
        assert cfg.chunk_size == 999
