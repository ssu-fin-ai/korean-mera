"""LLM 에이전트 공통 기반 클래스 (Claude)

이 모듈은 Claude API 호출, JSON 파싱, RAG 패턴 포맷팅 등
모든 에이전트가 공통으로 사용하는 핵심 유틸리티를 제공한다.
"""

import json
import re
import time
from typing import Any

import anthropic
from loguru import logger

from config import ANTHROPIC_API_KEY, SETTINGS

# 싱글톤 패턴: API 클라이언트는 최초 1회만 생성해 재사용
_client = None


def _get_client() -> anthropic.Anthropic:
    """Anthropic 클라이언트 싱글톤 반환 (최초 호출 시 생성)"""
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


# settings.yaml에서 모델 파라미터 로드
MODEL = SETTINGS["llm"]["model"]
TEMPERATURE = SETTINGS["llm"]["temperature"]
MAX_TOKENS = SETTINGS["llm"]["max_tokens"]


def call_llm(system: str, user: str, retries: int = 3) -> str:
    """Claude API 호출 (지수 백오프 재시도 포함)

    API 오류 시 5초·10초·15초 대기 후 재시도.
    3회 모두 실패하면 빈 문자열 반환 (호출부가 빈값 처리 담당).
    """
    client = _get_client()

    for attempt in range(retries):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return resp.content[0].text
        except Exception as e:
            # 재시도 간격: 5초, 10초, 15초 (선형 증가)
            wait = 5 * (attempt + 1)
            logger.warning(f"Claude 호출 실패 (시도 {attempt+1}/3): {e}. {wait}초 대기")
            time.sleep(wait)

    logger.error("Claude 최종 실패, 빈 문자열 반환")
    return ""


def _extract_json_object(text: str) -> str:
    """중괄호 깊이 추적으로 첫 번째 완전한 JSON 객체 추출

    LLM이 JSON 앞뒤에 설명 텍스트를 붙이는 경우에 대비해
    '{' 에서 시작해 중첩 깊이가 0이 되는 '}' 까지를 정확히 잘라낸다.
    문자열 내부의 이스케이프된 따옴표나 중괄호는 무시한다.
    """
    start = text.find("{")
    if start == -1:
        return text  # JSON 없으면 원본 반환
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True  # 다음 문자는 이스케이프 대상
            continue
        if ch == '"':
            in_string = not in_string  # 문자열 진입/탈출 토글
            continue
        if in_string:
            continue  # 문자열 안의 { } 는 깊이 계산에서 제외
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]  # 완전한 JSON 객체 발견
    return text[start:]  # 닫는 괄호가 없으면 시작부터 끝까지 반환


def parse_json_response(text: str) -> dict[str, Any]:
    """LLM 응답 텍스트에서 JSON 블록 추출 후 파싱

    처리 순서:
    1. ```json...``` 마크다운 코드블록이 있으면 내용만 추출
    2. 중괄호 깊이 추적으로 완전한 JSON 객체 경계 탐색
    3. 첫 번째 파싱 시도 (json.loads)
    4. 실패 시 trailing comma 제거 후 재시도
    5. 최종 실패 시 빈 dict 반환
    """
    # LLM이 ```json ... ``` 형식으로 감싸서 응답하는 경우 처리
    block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if block:
        text = block.group(1)

    text = _extract_json_object(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # trailing comma 제거 (예: {"a": 1,} → {"a": 1})
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 실패: {text[:200]}")
            return {}


def format_retrieved_patterns(patterns: list[dict]) -> str:
    """유사 패턴 목록을 LLM 프롬프트용 텍스트로 변환

    ChromaDB에서 검색된 RAG 패턴 각각에 대해:
    - 유사도 (1 - cosine distance)
    - 5일 후 실제 수익률 (label_5d)
    - 패턴 텍스트
    를 번호 매겨 포맷팅한다. 패턴이 없으면 '유사 패턴 없음' 반환.
    """
    if not patterns:
        return "유사 패턴 없음"
    lines = []
    for i, p in enumerate(patterns, 1):
        meta = p.get("metadata", {})
        label = meta.get("label_5d", "N/A")
        # label_5d는 소수 (0.05 = +5%) → 퍼센트로 변환해 표시
        label_str = f"{float(label)*100:+.1f}%" if label not in ("N/A", "") else "N/A"
        sim = 1 - p.get("distance", 1)  # cosine distance → 유사도 (1=완전일치)
        lines.append(
            f"[사례{i}] 유사도:{sim:.2f} | 이후5일:{label_str}\n{p['text']}"
        )
    return "\n\n".join(lines)
