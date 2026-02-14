"""ChromaDB 管理器：按嵌入模型路由到独立集合。"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from app.config import get_settings

COLLECTION_PREFIX = "doc_chunks__"


def _slugify_model(embed_model: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", embed_model.lower())
    return normalized.strip("_")


def build_collection_name(embed_model: str) -> str:
    """将嵌入模型名映射为集合名。"""
    return f"{COLLECTION_PREFIX}{_slugify_model(embed_model)}"


class ChromaManager:
    """向量库管理：每个嵌入模型一套集合。"""

    def __init__(self, client: Any | None = None):
        self._client = client or self._build_default_client()
        self._collection_cache: dict[str, Any] = {}

    @staticmethod
    def _build_default_client() -> Any:
        import chromadb

        settings = get_settings()
        return chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def get_collection(self, embed_model: str):
        """按嵌入模型获取（或创建）对应集合。"""
        collection_name = build_collection_name(embed_model)
        if collection_name in self._collection_cache:
            return self._collection_cache[collection_name]

        collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "embed_model": embed_model},
        )
        self._collection_cache[collection_name] = collection
        return collection

    def upsert_chunks(
        self,
        doc_id: int,
        chunks: Sequence[dict[str, Any]],
        embed_model: str,
        doc_hash: str,
    ) -> None:
        """将 chunk 批量写入目标模型集合。"""
        if not chunks:
            return

        collection = self.get_collection(embed_model)

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        embeddings: list[list[float]] = []
        has_embeddings = True

        for chunk in chunks:
            chunk_embedding = chunk.get("embedding")
            if chunk_embedding is None:
                has_embeddings = False
            else:
                embeddings.append(chunk_embedding)

            ids.append(str(chunk["chroma_id"]))
            documents.append(str(chunk["content"]))
            metadatas.append(
                {
                    "doc_id": doc_id,
                    "filename": chunk.get("filename"),
                    "file_type": chunk.get("file_type"),
                    "section": chunk.get("section"),
                    "chunk_index": chunk.get("chunk_index"),
                    "embed_model": embed_model,
                    "doc_hash": doc_hash,
                },
            )

        kwargs: dict[str, Any] = {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
        }
        if has_embeddings:
            kwargs["embeddings"] = embeddings

        collection.upsert(**kwargs)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        embed_model: str,
        where_extra: dict[str, Any] | None = None,
        n: int = 10,
    ) -> Any:
        """查询时按 embed_model 选择集合，避免跨向量空间混查。"""
        collection = self.get_collection(embed_model)
        return collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n,
            where=where_extra,
        )

    def _iter_model_collection_names(self) -> Iterable[str]:
        for collection in self._client.list_collections():
            name = getattr(collection, "name", None)
            if isinstance(name, str) and name.startswith(COLLECTION_PREFIX):
                yield name

    def delete_document(
        self,
        doc_id: int,
        *,
        across_all_models: bool = True,
        embed_model: str | None = None,
    ) -> None:
        """删除文档向量：默认遍历所有模型集合。"""
        if across_all_models:
            names = list(self._iter_model_collection_names())
            for name in names:
                collection = self._client.get_or_create_collection(name=name)
                collection.delete(where={"doc_id": doc_id})
            return

        if not embed_model:
            raise ValueError("across_all_models=False 时必须提供 embed_model")

        self.get_collection(embed_model).delete(where={"doc_id": doc_id})

    def rebuild_document_vectors(self, doc_id: int, *, target_embed_model: str) -> None:
        """向量重建占位接口（由上层 pipeline 编排）。"""
        raise NotImplementedError(
            f"rebuild_document_vectors(doc_id={doc_id}, target={target_embed_model}) 尚未实现",
        )
