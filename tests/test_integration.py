"""集成测试：验证完整启动流程"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


@pytest_asyncio.fixture
async def integration_engine() -> AsyncEngine:
    """集成测试专用引擎"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def integration_session(integration_engine: AsyncEngine) -> AsyncSession:
    """集成测试专用会话（含建表）"""
    from app.models.orm import Base

    async with integration_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    factory = async_sessionmaker(
        bind=integration_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session


class TestFullStartupFlow:
    """模拟应用启动的完整流程"""

    @pytest.mark.asyncio
    async def test_startup_sequence(self, integration_session: AsyncSession):
        """
        启动顺序：
        1. 加载静态配置
        2. 建表（已在 fixture 中完成）
        3. 加载动态配置
        4. 业务层可正常读取配置
        """
        from app.config import AppSettings, DynamicConfigPatch
        from app.services.config_service import DynamicConfigService

        settings = AppSettings(_env_file=None)
        assert settings.port == 8000

        svc = DynamicConfigService()
        cfg = await svc.load(integration_session)
        assert cfg.llm_provider == "deepseek"

        patch = DynamicConfigPatch(llm_provider="ollama", chunk_size=1500)
        updated = await svc.update(integration_session, patch)
        assert updated.llm_provider == "ollama"
        assert updated.chunk_size == 1500

        svc2 = DynamicConfigService()
        reloaded = await svc2.load(integration_session)
        assert reloaded.llm_provider == "ollama"
        assert reloaded.chunk_size == 1500

    @pytest.mark.asyncio
    async def test_document_crud_flow(self, integration_session: AsyncSession):
        """文档 CRUD 流程验证"""
        from app.models.orm import Chunk, Document, Entity
        from app.models.schemas import DocumentDetailResponse, DocumentResponse

        doc = Document(
            filename="集成测试.docx",
            file_type="docx",
            file_path="uploads/集成测试.docx",
            content_hash="integration_hash_" + "0" * 48,
            file_size=2048,
        )
        integration_session.add(doc)
        await integration_session.commit()
        await integration_session.refresh(doc)

        for i in range(3):
            chunk = Chunk(
                doc_id=doc.id,
                chunk_index=i,
                content=f"第 {i} 段文本内容",
                chroma_id=f"{doc.id}_{i}",
            )
            integration_session.add(chunk)
        doc.chunk_count = 3
        await integration_session.commit()

        entity = Entity(
            doc_id=doc.id,
            entity_type="person",
            entity_value="李四",
            normalized_value="李四",
            confidence=0.92,
        )
        integration_session.add(entity)
        await integration_session.commit()

        await integration_session.refresh(doc, ["chunks", "entities"])

        resp = DocumentResponse.model_validate(doc)
        assert resp.filename == "集成测试.docx"
        assert resp.chunk_count == 3

        detail = DocumentDetailResponse.model_validate(doc)
        assert len(detail.entities) == 1
        assert detail.entities[0].entity_value == "李四"

    @pytest.mark.asyncio
    async def test_template_extraction_flow(self, integration_session: AsyncSession):
        """模板 → 抽取结果流程验证"""
        from sqlalchemy import select

        from app.models.orm import Extraction, Template
        from app.models.schemas import TemplateUploadResponse

        tpl = Template(
            filename="报告模板.xlsx",
            file_type="xlsx",
            file_path="templates/报告模板.xlsx",
            field_count=3,
        )
        integration_session.add(tpl)
        await integration_session.commit()
        await integration_session.refresh(tpl)

        resp = TemplateUploadResponse.model_validate(tpl)
        assert resp.field_count == 3

        for name in ["项目名称", "负责人", "金额"]:
            ext = Extraction(template_id=tpl.id, field_name=name)
            integration_session.add(ext)
        await integration_session.commit()

        result = await integration_session.execute(
            select(Extraction).where(Extraction.template_id == tpl.id),
        )
        assert len(result.scalars().all()) == 3
