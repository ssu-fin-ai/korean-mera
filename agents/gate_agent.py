"""GateNet: 현재 패턴을 분석해 적합한 전문가 에이전트를 선택"""

from loguru import logger

from agents.base import call_llm, format_retrieved_patterns, parse_json_response

SYSTEM = """당신은 주식 패턴 분류 전문가입니다.
현재 종목의 상태와 유사한 과거 사례를 분석하여
가장 적합한 투자 전문가 유형을 선택합니다.
반드시 JSON 형식으로만 응답하세요."""

PROMPT_TEMPLATE = """## 현재 종목 상태
{current_text}

## 유사한 과거 사례 (Top-{k})
{retrieved_text}

## 선택 가능한 전문가
- growth   : 성장주 전문가 — 실적 모멘텀, 고PER, 성장 가속
- value    : 가치주 전문가 — 저PBR/PER, 안정 현금흐름, 자산가치
- theme    : 테마주 전문가 — 정책수혜, 이슈 급등, 단기 모멘텀
- dividend : 배당주 전문가 — 고배당, 방어적 섹터, 안정성
- crisis   : 위기종목 전문가 — 급락, 손절 판단, 반등 가능성

## 지시
1. 현재 패턴과 과거 사례를 종합 분석하세요.
2. 가장 적합한 전문가를 1~2개 선택하세요.
3. 아래 JSON 형식으로만 응답하세요.

출력 형식:
{{
  "experts": ["growth"],
  "confidence": 0.82,
  "pattern_type": "실적모멘텀 성장주",
  "reason": "RSI 65로 강세, 거래량 2배 급증, 20일 수익률 +12%로 모멘텀 강함"
}}"""


def route(
    current_text: str,
    retrieved_patterns: list[dict],
    max_experts: int = 2,
) -> dict:
    """
    Returns:
        {
          "experts": ["growth"],
          "confidence": 0.82,
          "pattern_type": "...",
          "reason": "..."
        }
    """
    retrieved_text = format_retrieved_patterns(retrieved_patterns)
    prompt = PROMPT_TEMPLATE.format(
        current_text=current_text,
        retrieved_text=retrieved_text,
        k=len(retrieved_patterns),
    )

    raw = call_llm(SYSTEM, prompt)
    result = parse_json_response(raw)

    if not result or "experts" not in result:
        logger.warning(f"GateNet 파싱 실패, 기본값 사용: {raw[:100]}")
        result = {"experts": ["growth"], "confidence": 0.5,
                  "pattern_type": "미분류", "reason": "파싱 실패"}

    # 최대 전문가 수 제한
    result["experts"] = result["experts"][:max_experts]
    return result
