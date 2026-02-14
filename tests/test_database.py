"""测试数据库连接管理"""

from unittest.mock import patch

import pytest
import pytest_asyncio


class TestGetEngine:
    """验证引擎创建"""

    def test_get_engine_returns_async_engine(self):
        """get_engine() 应返回 AsyncEngine"""
        from sqlalchemy.ext.asyncio import AsyncEngine

        with patch("app.config.get_settings") as mock:
            mock.return_value.database_url = "sqlite+aiosqlite:///:memory:"
            mock.return_value.log_level = "INFO"

            import app.models.database as db_mod

            db_mod._engine = None
            db_mod._session_factory = None

            engine = db_mod.get_engine()
            assert isinstance(engine, AsyncEngine)

            db_mod._engine = None
            db_mod._session_factory = None

    def test_get_engine_singleton(self):
        """get_engine() 应返回同一实例"""
        with patch("app.config.get_settings") as mock:
            mock.return_value.database_url = "sqlite+aiosqlite:///:memory:"
            mock.return_value.log_level = "INFO"

            import app.models.database as db_mod

            db_mod._engine = None
            db_mod._session_factory = None

            e1 = db_mod.get_engine()
            e2 = db_mod.get_engine()
            assert e1 is e2

            db_mod._engine = None
            db_mod._session_factory = None


class TestGetSessionFactory:
    """验证会话工厂"""

    def test_returns_session_factory(self):
        """get_session_factory() 应返回 async_sessionmaker"""
        from sqlalchemy.ext.asyncio import async_sessionmaker

        with patch("app.config.get_settings") as mock:
            mock.return_value.database_url = "sqlite+aiosqlite:///:memory:"
            mock.return_value.log_level = "INFO"

            import app.models.database as db_mod

            db_mod._engine = None
            db_mod._session_factory = None

            factory = db_mod.get_session_factory()
            assert isinstance(factory, async_sessionmaker)

            db_mod._engine = None
            db_mod._session_factory = None


class TestInitDb:
    """验证数据库初始化"""

    @pytest.mark.asyncio
    async def test_init_db_creates_tables(self):
        """init_db() 应创建全部表"""
        from sqlalchemy import text

        with patch("app.config.get_settings") as mock:
            mock.return_value.database_url = "sqlite+aiosqlite:///:memory:"
            mock.return_value.log_level = "INFO"

            import app.models.database as db_mod

            db_mod._engine = None
            db_mod._session_factory = None

            await db_mod.init_db()

            engine = db_mod.get_engine()
            async with engine.connect() as conn:
                result = await conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'"),
                )
                tables = {row[0] for row in result.fetchall()}

            expected = {
                "documents",
                "chunks",
                "entities",
                "templates",
                "extractions",
                "settings",
            }
            assert expected.issubset(tables)

            await db_mod.close_db()


class TestCloseDb:
    """验证关闭清理"""

    @pytest.mark.asyncio
    async def test_close_db_resets_state(self):
        """close_db() 应重置引擎和工厂"""
        with patch("app.config.get_settings") as mock:
            mock.return_value.database_url = "sqlite+aiosqlite:///:memory:"
            mock.return_value.log_level = "INFO"

            import app.models.database as db_mod

            db_mod._engine = None
            db_mod._session_factory = None

            db_mod.get_engine()
            assert db_mod._engine is not None

            await db_mod.close_db()
            assert db_mod._engine is None
            assert db_mod._session_factory is None


class TestGetDb:
    """验证 FastAPI 依赖注入"""

    @pytest.mark.asyncio
    async def test_get_db_yields_session(self):
        """get_db() 应 yield AsyncSession"""
        from sqlalchemy.ext.asyncio import AsyncSession

        with patch("app.config.get_settings") as mock:
            mock.return_value.database_url = "sqlite+aiosqlite:///:memory:"
            mock.return_value.log_level = "INFO"

            import app.models.database as db_mod

            db_mod._engine = None
            db_mod._session_factory = None

            async for session in db_mod.get_db():
                assert isinstance(session, AsyncSession)
                break

            await db_mod.close_db()
