"""测试 ChromaManager：每个嵌入模型使用独立集合。"""

from dataclasses import dataclass


@dataclass
class _CollectionView:
    name: str


class FakeCollection:
    def __init__(self, name: str):
        self.name = name
        self.upsert_calls: list[dict] = []
        self.query_calls: list[dict] = []
        self.delete_calls: list[dict] = []

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)

    def query(self, **kwargs):
        self.query_calls.append(kwargs)
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def delete(self, **kwargs):
        self.delete_calls.append(kwargs)


class FakeClient:
    def __init__(self):
        self.collections: dict[str, FakeCollection] = {}
        self.get_or_create_calls: list[tuple[str, dict | None]] = []

    def get_or_create_collection(self, name: str, metadata: dict | None = None):
        self.get_or_create_calls.append((name, metadata))
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]

    def list_collections(self):
        return [_CollectionView(name=name) for name in self.collections]


def test_build_collection_name_uses_model_slug():
    from app.vector_store.chroma_manager import build_collection_name

    got = build_collection_name("bge-large-zh-v1.5")
    assert got == "doc_chunks__bge_large_zh_v1_5"


def test_get_collection_routes_by_embed_model():
    from app.vector_store.chroma_manager import ChromaManager

    client = FakeClient()
    manager = ChromaManager(client=client)

    c1 = manager.get_collection("deepseek-embedding")
    c2 = manager.get_collection("bge-large-zh-v1.5")

    assert c1.name == "doc_chunks__deepseek_embedding"
    assert c2.name == "doc_chunks__bge_large_zh_v1_5"
    assert c1 is not c2


def test_query_routes_to_model_collection_and_passes_where():
    from app.vector_store.chroma_manager import ChromaManager

    client = FakeClient()
    manager = ChromaManager(client=client)

    manager.query(
        query_embedding=[0.1, 0.2],
        embed_model="deepseek-embedding",
        where_extra={"doc_id": 7},
        n=5,
    )

    deepseek_collection = client.collections["doc_chunks__deepseek_embedding"]
    assert len(deepseek_collection.query_calls) == 1
    assert deepseek_collection.query_calls[0]["query_embeddings"] == [[0.1, 0.2]]
    assert deepseek_collection.query_calls[0]["n_results"] == 5
    assert deepseek_collection.query_calls[0]["where"] == {"doc_id": 7}


def test_upsert_chunks_writes_to_target_model_collection():
    from app.vector_store.chroma_manager import ChromaManager

    client = FakeClient()
    manager = ChromaManager(client=client)

    chunks = [
        {
            "chroma_id": "1_0",
            "content": "第一段",
            "chunk_index": 0,
            "filename": "a.docx",
            "file_type": "docx",
            "section": "一",
            "embedding": [0.01, 0.02],
        },
        {
            "chroma_id": "1_1",
            "content": "第二段",
            "chunk_index": 1,
            "filename": "a.docx",
            "file_type": "docx",
            "section": "二",
            "embedding": [0.03, 0.04],
        },
    ]

    manager.upsert_chunks(
        doc_id=1,
        chunks=chunks,
        embed_model="deepseek-embedding",
        doc_hash="hash-001",
    )

    collection = client.collections["doc_chunks__deepseek_embedding"]
    assert len(collection.upsert_calls) == 1
    call = collection.upsert_calls[0]
    assert call["ids"] == ["1_0", "1_1"]
    assert call["documents"] == ["第一段", "第二段"]
    assert call["metadatas"][0]["doc_id"] == 1
    assert call["metadatas"][0]["embed_model"] == "deepseek-embedding"
    assert call["metadatas"][0]["doc_hash"] == "hash-001"


def test_delete_document_scans_all_model_collections():
    from app.vector_store.chroma_manager import ChromaManager

    client = FakeClient()
    manager = ChromaManager(client=client)

    manager.get_collection("deepseek-embedding")
    manager.get_collection("bge-large-zh-v1.5")
    client.get_or_create_collection("other_collection", metadata=None)

    manager.delete_document(doc_id=9, across_all_models=True)

    c1 = client.collections["doc_chunks__deepseek_embedding"]
    c2 = client.collections["doc_chunks__bge_large_zh_v1_5"]
    c_other = client.collections["other_collection"]
    assert c1.delete_calls == [{"where": {"doc_id": 9}}]
    assert c2.delete_calls == [{"where": {"doc_id": 9}}]
    assert c_other.delete_calls == []
