"""OpenAI 임베딩 (text-embedding-3-small, 1536차원)

패턴 텍스트를 벡터로 변환해 ChromaDB에 저장하거나
유사 패턴 검색 시 쿼리 벡터를 생성한다.

text-embedding-3-small: OpenAI의 경량 임베딩 모델.
1536차원 벡터, 비용 효율적이며 한국어도 잘 처리한다.
"""

from typing import List

from loguru import logger

from config import OPENAI_API_KEY, SETTINGS

# settings.yaml에서 임베딩 모델명 로드 (text-embedding-3-small)
_MODEL_NAME = SETTINGS["llm"]["embedding_model"]
EMBED_DIM = 1536  # text-embedding-3-small 고정 차원 수

# 싱글톤 OpenAI 클라이언트 (최초 호출 시 생성)
_client = None


def _get_client():
    """OpenAI 클라이언트 싱글톤 반환 (지연 초기화)"""
    global _client
    if _client is None:
        import openai
        _client = openai.OpenAI(api_key=OPENAI_API_KEY)
    return _client


def embed_texts(texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """텍스트 목록 → 임베딩 벡터 목록 (OpenAI API 배치 호출)

    batch_size개씩 나눠 API 호출 (단일 요청 최대 크기 제한 대응).
    빈 문자열은 "데이터 없음"으로 대체 (API 오류 방지).
    반환: [[float, ...], ...] - 입력 순서와 동일한 1536차원 벡터 목록
    """
    client = _get_client()
    # 빈 문자열이 있으면 API가 오류를 낼 수 있으므로 대체 텍스트 사용
    clean = [t if t.strip() else "데이터 없음" for t in texts]

    results: List[List[float]] = []
    for i in range(0, len(clean), batch_size):
        batch = clean[i:i + batch_size]
        resp = client.embeddings.create(model=_MODEL_NAME, input=batch)
        # resp.data는 Embedding 객체 목록; .embedding이 실제 벡터
        results.extend(item.embedding for item in resp.data)
        if len(clean) > batch_size:
            logger.debug(f"임베딩 진행: {min(i + batch_size, len(clean))}/{len(clean)}")

    return results


def embed_single(text: str) -> List[float]:
    """단일 텍스트 → 임베딩 벡터 (embed_texts의 편의 래퍼)"""
    return embed_texts([text])[0]
