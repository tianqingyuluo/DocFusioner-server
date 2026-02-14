"""测试 ORM 模型定义"""

import pytest
import pytest_asyncio
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession


class TestORMTableCreation:
    """验证所有表可被正确创建"""

    @pytest.mark.asyncio
    async def test_all_tables_created(self, async_engine: AsyncEngine):
        """验证 Base.metadata.create_all 创建全部 6 张表"""
        from app.models.orm import Base

        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        expected_tables = {
            "documents",
            "chunks",
            "entities",
            "templates",
            "extractions",
            "settings",
        }

        async with async_engine.connect() as conn:
            result = await conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'"),
            )
            actual_tables = {row[0] for row in result.fetchall()}

        assert expected_tables.issubset(actual_tables), (
            f"缺少表: {expected_tables - actual_tables}"
        )


class TestDocumentModel:
    """验证 Document ORM 模型"""

    @pytest.mark.asyncio
    async def test_create_document(self, db_session: AsyncSession):
        """可以创建并读取 Document"""
        from app.models.orm import Document

        doc = Document(
            filename="test.docx",
            file_type="docx",
            file_path="data/uploads/test.docx",
            content_hash="abc123" * 10 + "abcd",
        )
        db_session.add(doc)
        await db_session.commit()
        await db_session.refresh(doc)

        assert doc.id is not None
        assert doc.id > 0
        assert doc.status == "pending"
        assert doc.chunk_count == 0

    @pytest.mark.asyncio
    async def test_content_hash_unique(self, db_session: AsyncSession):
        """content_hash 唯一约束"""
        from sqlalchemy.exc import IntegrityError

        from app.models.orm import Document

        hash_val = "unique_hash_" + "0" * 52
        doc1 = Document(
            filename="a.docx",
            file_type="docx",
            file_path="a.docx",
            content_hash=hash_val,
        )
        doc2 = Document(
            filename="b.docx",
            file_type="docx",
            file_path="b.docx",
            content_hash=hash_val,
        )
        db_session.add(doc1)
        await db_session.commit()

        db_session.add(doc2)
        with pytest.raises(IntegrityError):
            await db_session.commit()


class TestChunkModel:
    """验证 Chunk ORM 模型"""

    @pytest.mark.asyncio
    async def test_create_chunk_with_document(self, db_session: AsyncSession):
        """Chunk 通过外键关联 Document"""
        from app.models.orm import Chunk, Document

        doc = Document(
            filename="test.md",
            file_type="md",
            file_path="test.md",
            content_hash="hash_chunk_test_" + "0" * 49,
        )
        db_session.add(doc)
        await db_session.commit()
        await db_session.refresh(doc)

        chunk = Chunk(
            doc_id=doc.id,
            chunk_index=0,
            content="Hello, world!",
            chroma_id=f"{doc.id}_0",
        )
        db_session.add(chunk)
        await db_session.commit()
        await db_session.refresh(chunk)

        assert chunk.id is not None
        assert chunk.vector_status == "pending"
        assert chunk.doc_id == doc.id

    @pytest.mark.asyncio
    async def test_unique_doc_chunk_index(self, db_session: AsyncSession):
        """(doc_id, chunk_index) 唯一约束"""
        from sqlalchemy.exc import IntegrityError

        from app.models.orm import Chunk, Document

        doc = Document(
            filename="dup.md",
            file_type="md",
            file_path="dup.md",
            content_hash="hash_dup_test_" + "0" * 51,
        )
        db_session.add(doc)
        await db_session.commit()
        await db_session.refresh(doc)

        c1 = Chunk(
            doc_id=doc.id,
            chunk_index=0,
            content="a",
            chroma_id=f"{doc.id}_0_a",
        )
        c2 = Chunk(
            doc_id=doc.id,
            chunk_index=0,
            content="b",
            chroma_id=f"{doc.id}_0_b",
        )
        db_session.add(c1)
        await db_session.commit()

        db_session.add(c2)
        with pytest.raises(IntegrityError):
            await db_session.commit()

    @pytest.mark.asyncio
    async def test_cascade_delete(self, db_session: AsyncSession):
        """删除 Document 时级联删除 Chunks"""
        from sqlalchemy import select

        from app.models.orm import Chunk, Document

        doc = Document(
            filename="cascade.md",
            file_type="md",
            file_path="cascade.md",
            content_hash="hash_cascade_" + "0" * 51,
        )
        db_session.add(doc)
        await db_session.commit()
        await db_session.refresh(doc)

        chunk = Chunk(
            doc_id=doc.id,
            chunk_index=0,
            content="text",
            chroma_id=f"{doc.id}_0_cascade",
        )
        db_session.add(chunk)
        await db_session.commit()

        await db_session.delete(doc)
        await db_session.commit()

        result = await db_session.execute(select(Chunk))
        assert result.scalars().all() == []


class TestEntityModel:
    """验证 Entity ORM 模型"""

    @pytest.mark.asyncio
    async def test_create_entity(self, db_session: AsyncSession):
        """可以创建 Entity 并关联 Document"""
        from app.models.orm import Document, Entity

        doc = Document(
            filename="entity.docx",
            file_type="docx",
            file_path="entity.docx",
            content_hash="hash_entity_" + "0" * 52,
        )
        db_session.add(doc)
        await db_session.commit()
        await db_session.refresh(doc)

        entity = Entity(
            doc_id=doc.id,
            entity_type="person",
            entity_value="张三",
            normalized_value="张三",
            confidence=0.95,
        )
        db_session.add(entity)
        await db_session.commit()
        await db_session.refresh(entity)

        assert entity.id is not None
        assert entity.entity_type == "person"


class TestTemplateModel:
    """验证 Template ORM 模型"""

    @pytest.mark.asyncio
    async def test_create_template(self, db_session: AsyncSession):
        """可以创建 Template"""
        from app.models.orm import Template

        tpl = Template(
            filename="report.xlsx",
            file_type="xlsx",
            file_path="templates/report.xlsx",
        )
        db_session.add(tpl)
        await db_session.commit()
        await db_session.refresh(tpl)

        assert tpl.id is not None
        assert tpl.field_count == 0


class TestExtractionModel:
    """验证 Extraction ORM 模型"""

    @pytest.mark.asyncio
    async def test_create_extraction(self, db_session: AsyncSession):
        """可以创建 Extraction 并关联 Template"""
        from app.models.orm import Extraction, Template

        tpl = Template(
            filename="tpl.xlsx",
            file_type="xlsx",
            file_path="tpl.xlsx",
        )
        db_session.add(tpl)
        await db_session.commit()
        await db_session.refresh(tpl)

        ext = Extraction(
            template_id=tpl.id,
            field_name="项目名称",
            field_value="DocFusion",
            confidence=0.88,
        )
        db_session.add(ext)
        await db_session.commit()
        await db_session.refresh(ext)

        assert ext.id is not None
        assert ext.field_value == "DocFusion"

    @pytest.mark.asyncio
    async def test_unique_template_field(self, db_session: AsyncSession):
        """(template_id, field_name) 唯一约束"""
        from sqlalchemy.exc import IntegrityError

        from app.models.orm import Extraction, Template

        tpl = Template(
            filename="uq.xlsx",
            file_type="xlsx",
            file_path="uq.xlsx",
        )
        db_session.add(tpl)
        await db_session.commit()
        await db_session.refresh(tpl)

        e1 = Extraction(template_id=tpl.id, field_name="字段A")
        e2 = Extraction(template_id=tpl.id, field_name="字段A")
        db_session.add(e1)
        await db_session.commit()

        db_session.add(e2)
        with pytest.raises(IntegrityError):
            await db_session.commit()


class TestSettingModel:
    """验证 Setting ORM 模型"""

    @pytest.mark.asyncio
    async def test_create_setting(self, db_session: AsyncSession):
        """Setting 使用 key 作为主键"""
        from sqlalchemy import select

        from app.models.orm import Setting

        s = Setting(key="llm_provider", value='"deepseek"')
        db_session.add(s)
        await db_session.commit()

        result = await db_session.execute(
            select(Setting).where(Setting.key == "llm_provider"),
        )
        setting = result.scalar_one()
        assert setting.value == '"deepseek"'

    @pytest.mark.asyncio
    async def test_merge_setting(self, db_session: AsyncSession):
        """merge 实现 upsert 语义"""
        from sqlalchemy import select

        from app.models.orm import Setting

        s1 = Setting(key="chunk_size", value="800")
        db_session.add(s1)
        await db_session.commit()

        s2 = Setting(key="chunk_size", value="1200")
        await db_session.merge(s2)
        await db_session.commit()

        result = await db_session.execute(
            select(Setting).where(Setting.key == "chunk_size"),
        )
        setting = result.scalar_one()
        assert setting.value == "1200"
