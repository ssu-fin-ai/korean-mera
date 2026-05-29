"""OpenAI 임베딩 (text-embedding-3-small, 1536차원)"""

from typing import List

from loguru import logger

from config import OPENAI_API_KEY, SETTINGS

_MODEL_NAME = SETTINGS["llm"]["embedding_model"]
EMBED_DIM = 1536  # text-embedding-3-small 차원

_client = None


def _get_client():
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """텍스트 목록 → 임베딩 벡터 목록 (OpenAI API)"""
    client = _get_client()
    clean = [t if t.strip() else "데이터 없음" for t in texts]

    results: List[List[float]] = []
    for i in range(0, len(clean), batch_size):
        batch = clean[i:i + batch_size]
        resp = client.embeddings.create(model=_MODEL_NAME, input=batch)
        results.extend(item.embedding for item in resp.data)
        if len(clean) > batch_size:
            logger.debug(f"임베딩 진행: {min(i + batch_size, len(clean))}/{len(clean)}")

    return results


def embed_single(text: str) -> List[float]:
    return embed_texts([text])[0]
