"""LLM 에이전트 공통 기반 클래스 (Gemini Flash)"""

import json
import re
import time
from typing import Any

from google import genai
from google.genai import types
from loguru import logger

from config import GOOGLE_API_KEY, SETTINGS

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GOOGLE_API_KEY)
    return _client


MODEL = SETTINGS["llm"]["model"]
TEMPERATURE = SETTINGS["llm"]["temperature"]
MAX_TOKENS = SETTINGS["llm"]["max_tokens"]


def _extract_text(resp) -> str:
    """Gemini 응답에서 텍스트 추출 (모델 버전별 구조 차이 대응)"""
    # 2.5+ 모델: thinking 포함 구조
    if resp.text is not None:
        return resp.text
    try:
        for part in resp.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                return part.text
    except Exception:
        pass
    return ""


def call_llm(system: str, user: str, retries: int = 3) -> str:
    """Gemini Flash API 호출 with 재시도"""
    client = _get_client()
    prompt = f"{system}\n\n{user}"

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=TEMPERATURE,
                    max_output_tokens=MAX_TOKENS,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return _extract_text(resp)
        except Exception as e:
            wait = 5 * (attempt + 1)
            logger.warning(f"Gemini 호출 실패 (시도 {attempt+1}/3): {e}. {wait}초 대기")
            time.sleep(wait)

    logger.error("Gemini 최종 실패, 빈 문자열 반환")
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
    # 마크다운 코드블록 안의 텍스트만 추출
    block = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if block:
        text = block.group(1)

    text = _extract_json_object(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # trailing comma 등 흔한 LLM 출력 오류 정리 후 재시도
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
