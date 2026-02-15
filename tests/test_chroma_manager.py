"""测试 ChromaManager：多集合路由 + typed 接口 + 分批 upsert。"""

import logging
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from app.vector_store.types import ChunkData, ChunkMetadata, QueryHit, QueryResult, UpsertResult


@dataclass
class _CollectionView:
    name: str


class FakeCollection:
    """模拟 Chroma Collection"""

    def __init__(self, name: str):
        self.name = name
        self.upsert_calls: list[dict] = []
        self.query_calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self._fail_next_upsert = False
        self._count = 0

    def upsert(self, **kwargs):
        if self._fail_next_upsert:
            self._fail_next_upsert = False
            raise RuntimeError("模拟写入失败")
        self.upsert_calls.append(kwargs)
        self._count += len(kwargs.get("ids", []))

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {
            "ids": [["1_0", "1_1"]],
            "documents": [["文本一", "文本二"]],
            "metadatas": [[{"doc_id": 1}, {"doc_id": 1}]],
            "distances": [[0.1, 0.3]],
        }

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)

    def count(self):
        return self._count


class FakeClient:
    """模拟 Chroma Client"""

    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}

    def get_or_create_collection(self, name: str, metadata: dict | None = None):
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]

    def list_collections(self):
        return [_CollectionView(name=name) for name in self.collections]


def _make_chunk(doc_id: int, index: int, section: str | None = None) -> ChunkData:
    """创建测试用 ChunkData"""
    return ChunkData(
        chroma_id=f"{doc_id}_{index}",
        content=f"内容块 {index}",
        embedding=[0.1 * index, 0.2 * index],
        metadata=ChunkMetadata(
            doc_id=doc_id,
            filename="test.docx",
            file_type="docx",
            chunk_index=index,
            doc_hash="testhash",
            section=section,
        ),
    )


class TestCollectionRouting:
    """多集合路由测试"""

    def test_build_collection_name(self):
        from app.vector_store.chroma_manager import build_collection_name

        assert build_collection_name("deepseek-embedding") == "doc_chunks__deepseek_embedding"
        assert build_collection_name("bge-large-zh-v1.5") == "doc_chunks__bge_large_zh_v1_5"

    def test_get_collection_creates_with_cosine(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)
        col = mgr.get_collection("deepseek-embedding")
        assert col.name == "doc_chunks__deepseek_embedding"

    def test_get_collection_caches(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)
        c1 = mgr.get_collection("model-a")
        c2 = mgr.get_collection("model-a")
        assert c1 is c2


class TestTypedUpsert:
    """typed upsert 测试"""

    def test_upsert_chunks_typed_input(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        chunks = [_make_chunk(1, 0, "第一章"), _make_chunk(1, 1)]
        result = mgr.upsert_chunks(chunks, embed_model="deepseek-embedding")

        assert isinstance(result, UpsertResult)
        assert result.success_ids == ["1_0", "1_1"]
        assert result.failed_ids == []

        col = client.collections["doc_chunks__deepseek_embedding"]
        call = col.upsert_calls[0]
        assert call["ids"] == ["1_0", "1_1"]
        assert call["documents"] == ["内容块 0", "内容块 1"]
        assert "section" in call["metadatas"][0]
        assert "section" not in call["metadatas"][1]

    def test_upsert_empty_chunks(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)
        result = mgr.upsert_chunks([], embed_model="deepseek-embedding")
        assert result.success_ids == []
        assert result.failed_ids == []


class TestBatchUpsert:
    """分批 upsert + 二分降批测试"""

    def test_batch_upsert_splits_large_input(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        chunks = [_make_chunk(1, i) for i in range(300)]
        result = mgr.upsert_chunks(chunks, embed_model="model-a", batch_size=256)

        assert len(result.success_ids) == 300
        col = client.collections["doc_chunks__model_a"]
        assert len(col.upsert_calls) == 2

    def test_batch_upsert_binary_split_on_failure(self):
        """失败时二分降批重试"""
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        col = client.collections.setdefault(
            "doc_chunks__model_a", FakeCollection("doc_chunks__model_a")
        )
        col._fail_next_upsert = True

        chunks = [_make_chunk(1, i) for i in range(4)]
        result = mgr.upsert_chunks(chunks, embed_model="model-a", batch_size=4)

        assert len(result.success_ids) == 4
        assert len(result.failed_ids) == 0


class TestTypedQuery:
    """typed query 测试"""

    def test_query_returns_typed_result(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        result = mgr.query(
            query_embedding=[0.1, 0.2],
            embed_model="deepseek-embedding",
            n_results=5,
        )

        assert isinstance(result, QueryResult)
        assert len(result.results) == 2
        assert isinstance(result.results[0], QueryHit)
        assert result.results[0].chroma_id == "1_0"
        assert result.results[0].distance == 0.1

    def test_query_with_doc_id_filter(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        mgr.query(
            query_embedding=[0.1],
            embed_model="model-a",
            doc_id=5,
        )

        col = client.collections["doc_chunks__model_a"]
        assert col.query_calls[0]["where"] == {"doc_id": 5}

    def test_query_with_doc_ids_filter(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        mgr.query(
            query_embedding=[0.1],
            embed_model="model-a",
            doc_ids=[1, 2, 3],
        )

        col = client.collections["doc_chunks__model_a"]
        assert col.query_calls[0]["where"] == {"doc_id": {"$in": [1, 2, 3]}}

    def test_query_no_filter_passes_none_where(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        mgr.query(query_embedding=[0.1], embed_model="model-a")

        col = client.collections["doc_chunks__model_a"]
        assert col.query_calls[0]["where"] is None

    def test_query_doc_ids_takes_priority_over_doc_id(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)

        mgr.query(
            query_embedding=[0.1],
            embed_model="model-a",
            doc_id=1,
            doc_ids=[2, 3],
        )

        col = client.collections["doc_chunks__model_a"]
        assert col.query_calls[0]["where"] == {"doc_id": {"$in": [2, 3]}}


class TestDelete:
    """删除测试"""

    def test_delete_by_doc_id_single_model(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)
        mgr.get_collection("model-a")

        mgr.delete_by_doc_id(doc_id=5, embed_model="model-a")

        col = client.collections["doc_chunks__model_a"]
        assert col.delete_calls == [{"where": {"doc_id": 5}}]

    def test_delete_by_doc_id_across_all(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)
        mgr.get_collection("model-a")
        mgr.get_collection("model-b")

        mgr.delete_by_doc_id(doc_id=9, across_all_models=True)

        for name in ["doc_chunks__model_a", "doc_chunks__model_b"]:
            col = client.collections[name]
            assert col.delete_calls == [{"where": {"doc_id": 9}}]


class TestCount:
    """count 测试"""

    def test_count_delegates(self):
        from app.vector_store.chroma_manager import ChromaManager

        client = FakeClient()
        mgr = ChromaManager(client=client)
        n = mgr.count("model-a")
        assert isinstance(n, int)
