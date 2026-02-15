"""测试向量库 typed 数据结构。"""

from app.vector_store.types import (
    ChunkData,
    ChunkMetadata,
    QueryHit,
    QueryResult,
    UpsertResult,
)


class TestChunkMetadata:
    """ChunkMetadata dataclass 测试"""

    def test_required_fields(self):
        meta = ChunkMetadata(
            doc_id=1,
            filename="test.docx",
            file_type="docx",
            chunk_index=0,
            doc_hash="abc123",
        )
        assert meta.doc_id == 1
        assert meta.filename == "test.docx"

    def test_section_default_none(self):
        meta = ChunkMetadata(
            doc_id=1, filename="t.txt", file_type="txt", chunk_index=0, doc_hash="h"
        )
        assert meta.section is None

    def test_to_chroma_dict_skips_none(self):
        meta = ChunkMetadata(
            doc_id=1,
            filename="test.md",
            file_type="md",
            chunk_index=0,
            doc_hash="hash",
            section=None,
        )
        d = meta.to_chroma_dict()
        assert "section" not in d
        assert d["doc_id"] == 1
        assert d["filename"] == "test.md"

    def test_to_chroma_dict_includes_section_when_set(self):
        meta = ChunkMetadata(
            doc_id=1,
            filename="test.docx",
            file_type="docx",
            chunk_index=0,
            doc_hash="hash",
            section="第一章",
        )
        d = meta.to_chroma_dict()
        assert d["section"] == "第一章"


class TestChunkData:
    """ChunkData dataclass 测试"""

    def test_all_fields(self):
        meta = ChunkMetadata(
            doc_id=1, filename="t.txt", file_type="txt", chunk_index=0, doc_hash="h"
        )
        chunk = ChunkData(
            chroma_id="1_0",
            content="hello",
            embedding=[0.1, 0.2, 0.3],
            metadata=meta,
        )
        assert chunk.chroma_id == "1_0"
        assert chunk.embedding == [0.1, 0.2, 0.3]


class TestUpsertResult:
    """UpsertResult dataclass 测试"""

    def test_defaults_empty(self):
        result = UpsertResult()
        assert result.success_ids == []
        assert result.failed_ids == []

    def test_with_values(self):
        result = UpsertResult(success_ids=["1_0", "1_1"], failed_ids=["1_2"])
        assert len(result.success_ids) == 2
        assert len(result.failed_ids) == 1


class TestQueryHit:
    """QueryHit dataclass 测试"""

    def test_all_fields(self):
        hit = QueryHit(
            chroma_id="1_0",
            content="hello",
            distance=0.15,
            metadata={"doc_id": 1, "filename": "test.txt"},
        )
        assert hit.distance == 0.15


class TestQueryResult:
    """QueryResult dataclass 测试"""

    def test_defaults_empty(self):
        result = QueryResult()
        assert result.results == []
