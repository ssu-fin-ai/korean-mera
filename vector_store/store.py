"""ChromaDB 기반 패턴 벡터 저장소"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from loguru import logger

from config import ROOT, SETTINGS

CHROMA_PATH = str(ROOT / SETTINGS["paths"]["chroma_db"])


def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


class PatternStore:
    """주가 패턴 임베딩 저장/검색"""

    COLLECTION = "stock_patterns"

    def __init__(self):
        self._client = _get_client()
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        self._col.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{k: (v if v is not None else "") for k, v in metadata.items()}],
        )

    def upsert_batch(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        clean_meta = [
            {k: (v if v is not None else "") for k, v in m.items()}
            for m in metadatas
        ]
        self._col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=clean_meta,
        )
        logger.debug(f"PatternStore upsert {len(ids)}건")

    def query(
        self,
        embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """유사 패턴 Top-K 검색 → [{ text, metadata, distance }]"""
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        result = self._col.query(**kwargs)
        items = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            items.append({"text": doc, "metadata": meta, "distance": dist})
        return items

    def count(self) -> int:
        return self._col.count()


class NewsStore:
    """DART 공시 / 뉴스 임베딩 저장/검색"""

    COLLECTION = "news_filings"

    def __init__(self):
        self._client = _get_client()
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        self._col.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[{k: (v if v is not None else "") for k, v in metadata.items()}],
        )

    def query_by_ticker(
        self,
        embedding: list[float],
        ticker: str,
        top_k: int = 3,
    ) -> list[dict]:
        result = self._col.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where={"ticker": ticker},
            include=["documents", "metadatas", "distances"],
        )
        items = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            items.append({"text": doc, "metadata": meta, "distance": dist})
        return items
