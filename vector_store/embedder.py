"""로컬 임베딩 (KR-SBERT) — OpenAI API 불필요, 완전 무료"""

from typing import List

from loguru import logger

# 한국어 특화 SBERT 모델 (CPU 동작, 768차원)
_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
EMBED_DIM = 768

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"KR-SBERT 모델 로딩: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """텍스트 목록 → 임베딩 벡터 목록 (로컬 CPU 실행)"""
    model = _get_model()
    # 빈 문자열 방지
    clean = [t if t.strip() else "데이터 없음" for t in texts]
    embeddings = model.encode(
        clean,
        batch_size=batch_size,
        show_progress_bar=len(clean) > 50,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def embed_single(text: str) -> List[float]:
    return embed_texts([text])[0]
