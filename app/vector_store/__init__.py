"""向量库管理包。"""

from app.vector_store.chroma_manager import ChromaManager, build_collection_name
from app.vector_store.types import (
    ChunkData,
    ChunkMetadata,
    QueryHit,
    QueryResult,
    UpsertResult,
)

__all__ = [
    "ChromaManager",
    "build_collection_name",
    "ChunkData",
    "ChunkMetadata",
    "QueryHit",
    "QueryResult",
    "UpsertResult",
]
