"""
向量库 typed 数据结构。

所有数据结构使用 @dataclass 定义。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChunkMetadata:
    """块级向量库存储元数据。"""

    doc_id: int
    filename: str
    file_type: str
    chunk_index: int
    doc_hash: str
    section: str | None = None

    def to_chroma_dict(self) -> dict[str, Any]:
        """转换为 Chroma metadata dict，跳过 None 值。"""
        d: dict[str, Any] = {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
            "doc_hash": self.doc_hash,
        }
        if self.section is not None:
            d["section"] = self.section
        return d


@dataclass
class ChunkData:
    """upsert 输入数据。"""

    chroma_id: str
    content: str
    embedding: list[float]
    metadata: ChunkMetadata


@dataclass
class UpsertResult:
    """upsert 操作结果。"""

    success_ids: list[str] = field(default_factory=list)
    failed_ids: list[str] = field(default_factory=list)


@dataclass
class QueryHit:
    """单条检索结果。"""

    chroma_id: str
    content: str
    distance: float
    metadata: dict[str, Any]


@dataclass
class QueryResult:
    """检索结果集。"""

    results: list[QueryHit] = field(default_factory=list)
