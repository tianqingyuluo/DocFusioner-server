"""
SQLite ORM 模型定义。

表结构关系：
    documents ──1:N──> chunks
        │
        └──1:N──> entities
    templates ──1:N──> extractions
    settings (独立 key-value 表)
"""

from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """基类。"""


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="docx/xlsx/md/txt/pdf",
    )
    file_size: Mapped[int | None] = mapped_column(Integer)
    file_path: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="相对路径",
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        comment="SHA-256 内容哈希（入库前必须计算）",
    )
    title: Mapped[str | None] = mapped_column(String(500), comment="LLM 提取的标题")
    summary: Mapped[str | None] = mapped_column(Text, comment="LLM 生成的摘要")
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        comment="pending/processing/completed/failed",
    )
    error_message: Mapped[str | None] = mapped_column(
        Text,
        comment="处理失败时的错误信息（Pipeline 写入）",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )
    entities: Mapped[list["Entity"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_docs_filename", "filename"),
        Index("idx_docs_created", "created_at"),
        Index("idx_docs_status", "status"),
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="原始文本（冗余，避免跨系统查询）",
    )
    section: Mapped[str | None] = mapped_column(String(255), comment="所属章节/Sheet名")
    chroma_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
        comment="{doc_id}_{chunk_index}",
    )
    token_count: Mapped[int | None] = mapped_column(Integer)
    embed_model: Mapped[str | None] = mapped_column(
        String(100),
        comment="生成向量的嵌入模型标识",
    )
    vector_status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
        comment="pending/ready/failed — 向量写入状态",
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunk_doc_index"),
        Index("idx_chunks_doc_id", "doc_id"),
        Index("idx_chunks_doc_section", "doc_id", "section"),
    )


class Entity(Base):
    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    doc_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
    )
    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        comment="person/org/date/amount/location",
    )
    entity_value: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="标准化值（空则存原值）",
    )
    source_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"),
    )
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    document: Mapped["Document"] = relationship(back_populates="entities")
    source_chunk: Mapped["Chunk | None"] = relationship()

    __table_args__ = (
        Index("idx_entities_type_normalized", "entity_type", "normalized_value"),
        Index("idx_entities_type_value", "entity_type", "entity_value"),
        Index("idx_entities_doc_id", "doc_id"),
    )


class Template(Base):
    __tablename__ = "templates"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="docx/xlsx",
    )
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    template_hash: Mapped[str | None] = mapped_column(
        String(32),
        unique=True,
        comment="MD5 去重",
    )
    field_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    extractions: Mapped[list["Extraction"]] = relationship(
        back_populates="template",
        cascade="all, delete-orphan",
    )


class Extraction(Base):
    __tablename__ = "extractions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    template_id: Mapped[int] = mapped_column(
        ForeignKey("templates.id", ondelete="CASCADE"),
        nullable=False,
    )
    field_name: Mapped[str] = mapped_column(String(255), nullable=False)
    field_value: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source_chunk_id: Mapped[int | None] = mapped_column(
        ForeignKey("chunks.id", ondelete="SET NULL"),
    )
    source_doc_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
    )
    evidence_json: Mapped[str | None] = mapped_column(
        Text,
        comment="JSON: [{chunk_id, score, rank, snippet}]",
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        comment="upsert 时由应用层写入 UTC",
    )

    template: Mapped["Template"] = relationship(back_populates="extractions")
    source_chunk: Mapped["Chunk | None"] = relationship()
    source_doc: Mapped["Document | None"] = relationship()

    __table_args__ = (
        UniqueConstraint("template_id", "field_name", name="uq_extraction_template_field"),
        Index("idx_extractions_template", "template_id"),
    )


class Setting(Base):
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="JSON 序列化存储",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        comment="由 DynamicConfigService 应用层写入 UTC 时间",
    )
