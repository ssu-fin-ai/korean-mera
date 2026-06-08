"""ChromaDB 기반 패턴 벡터 저장소

과거 주가 패턴(텍스트 + 임베딩 벡터 + 메타데이터)을 로컬 디스크에 영속 저장하고
코사인 유사도 기반 유사 패턴 검색을 제공한다.

ChromaDB HNSW 인덱스: 대용량 벡터의 근사 최근접 이웃(ANN) 검색에 특화.
cosine space: 벡터 방향(패턴 유형) 기준 유사도 측정.
"""

from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from loguru import logger

from config import ROOT, SETTINGS

# 로컬 디스크 ChromaDB 저장 경로 (settings.yaml의 paths.chroma_db)
CHROMA_PATH = str(ROOT / SETTINGS["paths"]["chroma_db"])


def _get_client() -> chromadb.PersistentClient:
    """ChromaDB 영속 클라이언트 생성 (anonymized_telemetry 비활성화)"""
    return chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False),
    )


class PatternStore:
    """주가 패턴 임베딩 저장/검색 클래스

    컬렉션 이름: "stock_patterns"
    각 문서: {id, document(텍스트), embedding(1536차원), metadata(ticker/date/label 등)}
    """

    COLLECTION = "stock_patterns"

    def __init__(self):
        self._client = _get_client()
        # 컬렉션이 없으면 생성, 있으면 기존 컬렉션 재사용
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},  # 코사인 유사도 기반 HNSW 인덱스
        )

    def upsert(
        self,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        """단일 패턴 저장 (doc_id 중복 시 업데이트)

        doc_id 형식: "{ticker}_{YYYY-MM-DD}" (예: "005930_2025-01-15")
        metadata의 None 값은 빈 문자열로 변환 (ChromaDB는 None 미지원)
        """
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
        """다수 패턴 배치 저장 (히스토리 DB 구축 시 사용)

        ids/texts/embeddings/metadatas는 동일 인덱스끼리 대응.
        None 값 정리 후 ChromaDB에 일괄 upsert.
        """
        # ChromaDB는 metadata 값으로 None을 허용하지 않아 빈 문자열로 치환
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
        """유사 패턴 Top-K 코사인 유사도 검색

        where: ChromaDB 메타데이터 필터 (예: {"ticker": "005930"})
        반환: [{"text": ..., "metadata": {...}, "distance": float}, ...]
          distance: 코사인 거리 (0=완전일치, 1=완전반대)
          유사도로 변환: similarity = 1 - distance
        """
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where  # 특정 종목/섹터로 검색 범위 제한

        result = self._col.query(**kwargs)
        # ChromaDB 결과는 [[...]] 형태이므로 [0]으로 첫 번째 쿼리 결과 추출
        items = []
        for doc, meta, dist in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ):
            items.append({"text": doc, "metadata": meta, "distance": dist})
        return items

    def count(self) -> int:
        """저장된 패턴 총 건수 반환"""
        return self._col.count()
