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


class PortfolioStore:
    """일별 포트폴리오 BUY 신호 저장/조회"""

    COLLECTION = "portfolio_history"

    def __init__(self):
        self._client = _get_client()
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, portfolio_date: str, portfolio_df) -> int:
        """포트폴리오 DataFrame → ChromaDB 저장 (종목별 1 doc)"""
        from vector_store.embedder import embed_texts

        ids, texts, metas = [], [], []
        for rank, row in portfolio_df.iterrows():
            ticker = row["ticker"]
            reasons = row.get("reasons", [])
            reasons_text = "; ".join(reasons) if reasons else ""
            text = (
                f"BUY 신호 {portfolio_date}: {ticker} {row['name']} [{row['sector']}]\n"
                f"신뢰도:{row['confidence']:.0%} 점수:{row['score']:.2f} "
                f"목표수익:{row['target_return']:+.1%}\n"
                f"패턴:{row.get('pattern_type', '')}\n"
                f"근거:{reasons_text}"
            )
            ids.append(f"portfolio_{portfolio_date}_{ticker}")
            texts.append(text)
            metas.append({
                "portfolio_date": portfolio_date,
                "ticker": ticker,
                "name": str(row["name"]),
                "sector": str(row["sector"]),
                "signal": str(row["signal"]),
                "score": float(row["score"]),
                "confidence": float(row["confidence"]),
                "target_return": float(row["target_return"]),
                "pattern_type": str(row.get("pattern_type", "")),
                "rank": int(rank),
            })

        if not ids:
            return 0

        embeddings = embed_texts(texts)
        self._col.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metas,
        )
        logger.info(f"PortfolioStore: {len(ids)}종목 저장 ({portfolio_date})")
        return len(ids)

    def get_by_date(self, portfolio_date: str) -> list[dict]:
        """특정 날짜 포트폴리오 전체 조회"""
        result = self._col.get(
            where={"portfolio_date": {"$eq": portfolio_date}},
            include=["documents", "metadatas"],
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

    def list_dates(self) -> list[str]:
        """저장된 포트폴리오 날짜 목록 (오름차순)"""
        result = self._col.get(include=["metadatas"])
        dates = {m["portfolio_date"] for m in result["metadatas"] if "portfolio_date" in m}
        return sorted(dates)

    def count(self) -> int:
        return self._col.count()


class EvaluationStore:
    """포트폴리오 성과 평가 결과 저장/조회"""

    COLLECTION = "portfolio_evaluations"

    def __init__(self):
        self._client = _get_client()
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def save_evaluation(
        self,
        portfolio_date: str,
        eval_date: str,
        stock_evals: list[dict],
        summary: str,
        avg_return: float,
        hit_rate: float,
    ) -> None:
        from vector_store.embedder import embed_texts

        # 요약 doc (날짜당 1개)
        summary_id = f"eval_summary_{portfolio_date}"
        summary_meta = {
            "portfolio_date": portfolio_date,
            "eval_date": eval_date,
            "doc_type": "summary",
            "avg_return": float(avg_return),
            "hit_rate": float(hit_rate),
            "stock_count": len(stock_evals),
        }

        # 종목별 평가 docs
        stock_ids, stock_texts, stock_metas = [], [], []
        for e in stock_evals:
            text = (
                f"평가 {portfolio_date}→{eval_date}: {e['ticker']} {e['name']}\n"
                f"실제:{e['actual_return']:+.1%} 목표:{e['target_return']:+.1%} "
                f"적중:{'O' if e['correct'] else 'X'}\n"
                f"원인:{e.get('miss_reason', '')}\n"
                f"교훈:{e.get('lesson', '')}"
            )
            stock_ids.append(f"eval_{portfolio_date}_{e['ticker']}")
            stock_texts.append(text)
            stock_metas.append({
                "portfolio_date": portfolio_date,
                "eval_date": eval_date,
                "doc_type": "stock",
                "ticker": str(e["ticker"]),
                "name": str(e["name"]),
                "actual_return": float(e["actual_return"]),
                "target_return": float(e["target_return"]),
                "correct": int(e["correct"]),  # ChromaDB: bool → int
                "confidence_score": float(e.get("confidence_score", 0.5)),
                "miss_reason": str(e.get("miss_reason", "")),
                "lesson": str(e.get("lesson", "")),
            })

        all_ids = [summary_id] + stock_ids
        all_texts = [summary] + stock_texts
        all_metas = [summary_meta] + stock_metas
        all_embs = embed_texts(all_texts)

        self._col.upsert(
            ids=all_ids,
            documents=all_texts,
            embeddings=all_embs,
            metadatas=all_metas,
        )
        logger.info(
            f"EvaluationStore: {portfolio_date} 평가 저장 "
            f"(수익:{avg_return:+.1%} 적중:{hit_rate:.0%})"
        )

    def get_summary_by_date(self, portfolio_date: str) -> dict | None:
        """날짜별 요약 평가 조회 (없으면 None)"""
        result = self._col.get(
            where={"$and": [
                {"portfolio_date": {"$eq": portfolio_date}},
                {"doc_type": {"$eq": "summary"}},
            ]},
            include=["documents", "metadatas"],
        )
        if not result["documents"]:
            return None
        return {"text": result["documents"][0], "metadata": result["metadatas"][0]}

    def get_stock_evals_by_date(self, portfolio_date: str) -> list[dict]:
        """날짜별 종목 평가 목록 조회"""
        result = self._col.get(
            where={"$and": [
                {"portfolio_date": {"$eq": portfolio_date}},
                {"doc_type": {"$eq": "stock"}},
            ]},
            include=["documents", "metadatas"],
        )
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]

    def get_recent_summaries(self, n: int = 10) -> list[dict]:
        """최근 N개 요약 평가 (날짜 내림차순)"""
        result = self._col.get(
            where={"doc_type": {"$eq": "summary"}},
            include=["documents", "metadatas"],
        )
        items = [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(result["documents"], result["metadatas"])
        ]
        items.sort(key=lambda x: x["metadata"].get("portfolio_date", ""), reverse=True)
        return items[:n]

    def count(self) -> int:
        return self._col.count()
