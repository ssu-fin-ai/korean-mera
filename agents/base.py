"""LLM 에이전트 공통 기반 클래스 (Claude)"""

import json
import re
import time
from typing import Any

import anthropic
from loguru import logger

from config import ANTHROPIC_API_KEY, SETTINGS

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


MODEL = SETTINGS["llm"]["model"]
TEMPERATURE = SETTINGS["llm"]["temperature"]
MAX_TOKENS = SETTINGS["llm"]["max_tokens"]


def call_llm(system: str, user: str, retries: int = 3) -> str:
    """Claude API 호출 with 재시도"""
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
            wait = 5 * (attempt + 1)
            logger.warning(f"Claude 호출 실패 (시도 {attempt+1}/3): {e}. {wait}초 대기")
            time.sleep(wait)

    logger.error("Claude 최종 실패, 빈 문자열 반환")
    return ""


def _extract_json_object(text: str) -> str:
    """중괄호 깊이 추적으로 첫 번째 완전한 JSON 객체 추출"""
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


def parse_json_response(text: str) -> dict[str, Any]:
    """LLM 응답에서 JSON 블록 추출"""
    block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if block:
        text = block.group(1)

    text = _extract_json_object(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"JSON 파싱 실패: {text[:200]}")
            return {}


def format_retrieved_patterns(patterns: list[dict]) -> str:
    """유사 패턴 목록을 LLM 프롬프트용 텍스트로 변환"""
    if not patterns:
        return "유사 패턴 없음"
    lines = []
    for i, p in enumerate(patterns, 1):
        meta = p.get("metadata", {})
        label = meta.get("label_5d", "N/A")
        label_str = f"{float(label)*100:+.1f}%" if label not in ("N/A", "") else "N/A"
        sim = 1 - p.get("distance", 1)
        lines.append(
            f"[사례{i}] 유사도:{sim:.2f} | 이후5일:{label_str}\n{p['text']}"
        )
    return "\n\n".join(lines)
