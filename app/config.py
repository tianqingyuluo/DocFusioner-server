"""
配置管理模块

双层配置架构：
- 静态配置 (AppSettings): 从 .env 文件加载，敏感信息（API Key）仅存于此
- 动态配置 (DynamicConfig): schema 默认值 + SQLite overrides，前端可修改
"""

from functools import lru_cache
from typing import Annotated, Literal

from annotated_types import Ge, Le
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """静态配置：从 .env 文件 + 环境变量加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM API 密钥（敏感，仅 .env）----
    deepseek_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    zhipu_api_key: str = ""

    # ---- 数据存储路径 ----
    database_url: str = "sqlite+aiosqlite:///data/structured.db"
    chroma_persist_dir: str = "data/chroma_db"
    upload_dir: str = "data/uploads"

    # ---- 服务配置 ----
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> AppSettings:
    """获取静态配置单例。"""
    return AppSettings()


class DynamicConfig(BaseModel):
    """动态配置 Schema：默认值源 + 字段约束。"""

    model_config = ConfigDict(extra="ignore")

    llm_provider: Literal["deepseek", "ollama"] = "deepseek"
    llm_model: str = Field(default="deepseek-chat", min_length=1)
    embedding_provider: Literal["deepseek", "zhipu", "local"] = "deepseek"
    embedding_model: str = Field(default="deepseek-embedding", min_length=1)
    chunk_size: Annotated[int, Ge(100), Le(4000)] = 800
    chunk_overlap: Annotated[int, Ge(0), Le(2000)] = 100


class DynamicConfigPatch(BaseModel):
    """PATCH 更新模型：所有字段 Optional，白名单锁死。"""

    llm_provider: Literal["deepseek", "ollama"] | None = None
    llm_model: str | None = Field(default=None, min_length=1)
    embedding_provider: Literal["deepseek", "zhipu", "local"] | None = None
    embedding_model: str | None = Field(default=None, min_length=1)
    chunk_size: Annotated[int, Ge(100), Le(4000)] | None = None
    chunk_overlap: Annotated[int, Ge(0), Le(2000)] | None = None
