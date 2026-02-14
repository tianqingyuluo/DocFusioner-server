"""测试配置管理模块"""

import os

import pytest
from pydantic import ValidationError


class TestAppSettings:
    """静态配置测试"""

    def test_default_values(self):
        """验证 AppSettings 的所有默认值"""
        from app.config import AppSettings

        settings = AppSettings(
            _env_file=None,  # 忽略 .env 文件，测试纯默认值
        )
        assert settings.deepseek_api_key == ""
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.zhipu_api_key == ""
        assert settings.database_url == "sqlite+aiosqlite:///data/structured.db"
        assert settings.chroma_persist_dir == "data/chroma_db"
        assert settings.upload_dir == "data/uploads"
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.cors_origins == ["http://localhost:5173"]
        assert settings.log_level == "INFO"

    def test_env_override(self, monkeypatch):
        """验证环境变量可覆盖默认值"""
        from app.config import AppSettings

        monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key-123")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        settings = AppSettings(_env_file=None)
        assert settings.deepseek_api_key == "test-key-123"
        assert settings.port == 9000
        assert settings.log_level == "DEBUG"

    def test_extra_fields_ignored(self):
        """验证 extra='ignore' 策略"""
        from app.config import AppSettings

        os.environ["UNKNOWN_FIELD_XYZ"] = "whatever"
        try:
            settings = AppSettings(_env_file=None)
            assert settings is not None
        finally:
            del os.environ["UNKNOWN_FIELD_XYZ"]

    def test_get_settings_returns_singleton(self):
        """验证 get_settings() 返回缓存实例"""
        from app.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2


class TestDynamicConfig:
    """动态配置 Schema 测试"""

    def test_default_values(self):
        """验证所有默认值"""
        from app.config import DynamicConfig

        cfg = DynamicConfig()
        assert cfg.llm_provider == "deepseek"
        assert cfg.llm_model == "deepseek-chat"
        assert cfg.embedding_provider == "deepseek"
        assert cfg.embedding_model == "deepseek-embedding"
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 100

    def test_override_values(self):
        """验证可通过 kwargs 覆盖"""
        from app.config import DynamicConfig

        cfg = DynamicConfig(llm_provider="ollama", chunk_size=1200)
        assert cfg.llm_provider == "ollama"
        assert cfg.chunk_size == 1200

    def test_invalid_provider_rejected(self):
        """非法 provider 应被拒绝"""
        from app.config import DynamicConfig

        with pytest.raises(ValidationError):
            DynamicConfig(llm_provider="invalid_provider")

    def test_chunk_size_boundaries(self):
        """chunk_size 边界校验：100 <= x <= 4000"""
        from app.config import DynamicConfig

        assert DynamicConfig(chunk_size=100).chunk_size == 100
        assert DynamicConfig(chunk_size=4000).chunk_size == 4000

        with pytest.raises(ValidationError):
            DynamicConfig(chunk_size=99)
        with pytest.raises(ValidationError):
            DynamicConfig(chunk_size=4001)

    def test_chunk_overlap_boundaries(self):
        """chunk_overlap 边界校验：0 <= x <= 2000"""
        from app.config import DynamicConfig

        assert DynamicConfig(chunk_overlap=0).chunk_overlap == 0
        assert DynamicConfig(chunk_overlap=2000).chunk_overlap == 2000

        with pytest.raises(ValidationError):
            DynamicConfig(chunk_overlap=-1)
        with pytest.raises(ValidationError):
            DynamicConfig(chunk_overlap=2001)

    def test_empty_model_name_rejected(self):
        """空模型名应被拒绝（min_length=1）"""
        from app.config import DynamicConfig

        with pytest.raises(ValidationError):
            DynamicConfig(llm_model="")

    def test_extra_fields_ignored(self):
        """extra='ignore' 策略"""
        from app.config import DynamicConfig

        cfg = DynamicConfig(unknown_field="ignored")
        assert not hasattr(cfg, "unknown_field")


class TestDynamicConfigPatch:
    """PATCH 更新模型测试"""

    def test_all_none_by_default(self):
        """所有字段默认 None"""
        from app.config import DynamicConfigPatch

        patch = DynamicConfigPatch()
        dump = patch.model_dump(exclude_none=True)
        assert dump == {}

    def test_partial_update(self):
        """仅传入部分字段"""
        from app.config import DynamicConfigPatch

        patch = DynamicConfigPatch(llm_provider="ollama", chunk_size=1500)
        dump = patch.model_dump(exclude_none=True)
        assert dump == {"llm_provider": "ollama", "chunk_size": 1500}

    def test_invalid_provider_rejected(self):
        """Patch 也应校验 Literal 约束"""
        from app.config import DynamicConfigPatch

        with pytest.raises(ValidationError):
            DynamicConfigPatch(llm_provider="bad")

    def test_invalid_chunk_size_rejected(self):
        """Patch 也应校验数值边界"""
        from app.config import DynamicConfigPatch

        with pytest.raises(ValidationError):
            DynamicConfigPatch(chunk_size=50)
