"""LLM Embedding API 호출 (OpenAI text-embedding-3-small)"""

import time
from typing import List

from loguru import logger
from openai import OpenAI

from config import OPENAI_API_KEY, SETTINGS

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


EMBED_MODEL = SETTINGS["llm"]["embedding_model"]
EMBED_DIM = 1536  # text-embedding-3-small 차원


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """텍스트 목록 → 임베딩 벡터 목록"""
    client = _get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        # 빈 문자열 방지
        batch = [t if t.strip() else "no data" for t in batch]

        for attempt in range(3):
            try:
                resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
                batch_emb = [item.embedding for item in resp.data]
                all_embeddings.extend(batch_emb)
                break
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(f"임베딩 실패 (시도 {attempt+1}/3): {e}. {wait}초 대기")
                time.sleep(wait)
        else:
            # 실패 시 영벡터로 패딩
            logger.error(f"임베딩 최종 실패, 영벡터 사용: {batch[:1]}")
            all_embeddings.extend([[0.0] * EMBED_DIM] * len(batch))

    return all_embeddings


def embed_single(text: str) -> List[float]:
    return embed_texts([text])[0]
