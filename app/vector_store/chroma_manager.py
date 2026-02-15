"""ChromaDB 管理器：按嵌入模型路由到独立集合，typed 接口。"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Sequence
from typing import Any

from app.config import get_settings
from app.vector_store.types import ChunkData, QueryHit, QueryResult, UpsertResult

logger = logging.getLogger(__name__)

COLLECTION_PREFIX = "doc_chunks__"
_WHERE_WHITELIST = {"file_type", "filename", "section"}


def _slugify_model(embed_model: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", embed_model.lower())
    return normalized.strip("_")


def build_collection_name(embed_model: str) -> str:
    """将嵌入模型名映射为集合名。"""
    return f"{COLLECTION_PREFIX}{_slugify_model(embed_model)}"


class ChromaManager:
    """向量库管理：每个嵌入模型一套集合，typed 接口。"""

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
        chunks: Sequence[ChunkData],
        embed_model: str,
        *,
        batch_size: int = 256,
    ) -> UpsertResult:
        """批量写入向量，typed 输入输出，分批 + 二分降批重试。"""
        if not chunks:
            return UpsertResult()

        collection = self.get_collection(embed_model)
        success_ids: list[str] = []
        failed_ids: list[str] = []

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self._upsert_batch(collection, batch, success_ids, failed_ids)

        return UpsertResult(success_ids=success_ids, failed_ids=failed_ids)

    def _upsert_batch(
        self,
        collection: Any,
        batch: Sequence[ChunkData],
        success_ids: list[str],
        failed_ids: list[str],
    ) -> None:
        """单批写入，失败时二分降批重试。"""
        ids = [c.chroma_id for c in batch]
        documents = [c.content for c in batch]
        embeddings = [c.embedding for c in batch]
        metadatas = [c.metadata.to_chroma_dict() for c in batch]

        try:
            collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
            success_ids.extend(ids)
        except Exception:
            if len(batch) == 1:
                logger.warning("向量写入失败: chroma_id=%s", batch[0].chroma_id, exc_info=True)
                failed_ids.append(batch[0].chroma_id)
            else:
                mid = len(batch) // 2
                self._upsert_batch(collection, batch[:mid], success_ids, failed_ids)
                self._upsert_batch(collection, batch[mid:], success_ids, failed_ids)

    def query(
        self,
        query_embedding: Sequence[float],
        *,
        embed_model: str,
        n_results: int = 10,
        doc_id: int | None = None,
        doc_ids: list[int] | None = None,
        extra_where: dict | None = None,
    ) -> QueryResult:
        """typed 语义检索，embed_model 用于路由集合。"""
        collection = self.get_collection(embed_model)

        where: dict[str, Any] = {}

        if doc_ids is not None:
            where["doc_id"] = {"$in": doc_ids}
        elif doc_id is not None:
            where["doc_id"] = doc_id

        if extra_where:
            for k, v in extra_where.items():
                if k in _WHERE_WHITELIST:
                    where[k] = v

        raw = collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=n_results,
            where=where if where else None,
        )

        hits: list[QueryHit] = []
        if raw.get("ids") and raw["ids"][0]:
            for i, cid in enumerate(raw["ids"][0]):
                hits.append(
                    QueryHit(
                        chroma_id=cid,
                        content=raw["documents"][0][i] if raw.get("documents") else "",
                        distance=raw["distances"][0][i] if raw.get("distances") else 0.0,
                        metadata=raw["metadatas"][0][i] if raw.get("metadatas") else {},
                    )
                )

        return QueryResult(results=hits)

    def delete_by_doc_id(
        self,
        doc_id: int,
        *,
        embed_model: str | None = None,
        across_all_models: bool = False,
    ) -> None:
        """按文档 ID 删除向量。"""
        if across_all_models:
            for name in self._iter_model_collection_names():
                collection = self._client.get_or_create_collection(name=name)
                collection.delete(where={"doc_id": doc_id})
            return

        if embed_model is None:
            raise ValueError("across_all_models=False 时必须提供 embed_model")

        self.get_collection(embed_model).delete(where={"doc_id": doc_id})

    def count(self, embed_model: str) -> int:
        """指定集合内向量数量。"""
        return self.get_collection(embed_model).count()

    def _iter_model_collection_names(self) -> Iterable[str]:
        for collection in self._client.list_collections():
            name = getattr(collection, "name", None)
            if isinstance(name, str) and name.startswith(COLLECTION_PREFIX):
                yield name
