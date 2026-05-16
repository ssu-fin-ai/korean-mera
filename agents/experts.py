"""5개 전문가 에이전트 (성장주 / 가치주 / 테마주 / 배당주 / 위기종목)"""

from agents.base import call_llm, format_retrieved_patterns, parse_json_response
from loguru import logger

# ── 공통 출력 형식 ──────────────────────────────────────────────────────────

_JSON_FORMAT = """
출력 형식 (JSON만, 다른 텍스트 없이):
{{
  "signal": "BUY",
  "confidence": 0.75,
  "target_return": 0.08,
  "horizon_days": 10,
  "score": 7,
  "risks": ["리스크1", "리스크2"],
  "reason": "한 줄 근거"
}}
signal 값: BUY | HOLD | SELL
confidence: 0.0 ~ 1.0
target_return: 예상 수익률 (소수, 예: 0.08 = +8%)
horizon_days: 투자 기간(일)
score: 매력도 1~10점"""

_SYSTEM_BASE = "당신은 한국 주식시장 전문 애널리스트입니다. 반드시 JSON만 응답하세요."


def _build_prompt(role_desc: str, analysis_guide: str,
                  current_text: str, retrieved_text: str,
                  news_text: str, financials: dict) -> str:
    fin_str = ""
    if financials:
        fin_str = f"\n## 최근 재무\n매출:{financials.get('revenue','N/A')} 영업이익:{financials.get('op_income','N/A')} 순이익:{financials.get('net_income','N/A')}"

    return f"""{role_desc}

## 현재 종목 상태
{current_text}

## 유사 과거 패턴 결과
{retrieved_text}

## 최근 공시/뉴스
{news_text or '없음'}{fin_str}

## 분석 요청
{analysis_guide}

{_JSON_FORMAT}"""


# ── 성장주 전문가 ────────────────────────────────────────────────────────────

_GROWTH_ROLE = "당신은 한국 성장주(고PER, 실적 모멘텀) 전문 애널리스트입니다."
_GROWTH_GUIDE = """다음 관점에서 분석하세요:
1. 실적 모멘텀 (매출/영업이익 성장세)
2. 주가 모멘텀 (RSI, MACD, 거래량 급증 여부)
3. 섹터 성장성 (AI, 반도체, 2차전지, 바이오 등)
4. 리스크 요인 (과매수 구간, 실적 미스 가능성)"""


def growth_expert(current_text: str, retrieved_patterns: list[dict],
                  news_text: str = "", financials: dict = None) -> dict:
    prompt = _build_prompt(
        _GROWTH_ROLE, _GROWTH_GUIDE,
        current_text, format_retrieved_patterns(retrieved_patterns),
        news_text, financials or {}
    )
    raw = call_llm(_SYSTEM_BASE, prompt)
    result = parse_json_response(raw)
    if not result:
        logger.warning(f"growth_expert 파싱 실패: {raw[:100]}")
        return _default_signal()
    result["expert"] = "growth"
    return result


# ── 가치주 전문가 ────────────────────────────────────────────────────────────

_VALUE_ROLE = "당신은 한국 가치주(저PBR/PER, 자산가치, 안정 현금흐름) 전문 애널리스트입니다."
_VALUE_GUIDE = """다음 관점에서 분석하세요:
1. 밸류에이션 매력도 (PBR, PER 섹터 대비)
2. 재무 안정성 (부채비율, 이자보상배율)
3. 배당 및 자본 환원 정책
4. 저평가 해소 촉매제 여부"""


def value_expert(current_text: str, retrieved_patterns: list[dict],
                 news_text: str = "", financials: dict = None) -> dict:
    prompt = _build_prompt(
        _VALUE_ROLE, _VALUE_GUIDE,
        current_text, format_retrieved_patterns(retrieved_patterns),
        news_text, financials or {}
    )
    raw = call_llm(_SYSTEM_BASE, prompt)
    result = parse_json_response(raw)
    if not result:
        logger.warning(f"value_expert 파싱 실패: {raw[:100]}")
        return _default_signal()
    result["expert"] = "value"
    return result


# ── 테마주 전문가 ────────────────────────────────────────────────────────────

_THEME_ROLE = "당신은 한국 테마주(정책수혜, 이슈 모멘텀, 단기 급등) 전문 애널리스트입니다."
_THEME_GUIDE = """다음 관점에서 분석하세요:
1. 관련 정책/이슈 강도와 지속성
2. 수혜 직접성 (간접 수혜 vs 직접 수혜)
3. 단기 과열 여부 (RSI 70+, 거래량 폭발)
4. 테마 사이클 위치 (초기/중기/말기)
※ 테마주 특성상 손절선 명시 필요"""


def theme_expert(current_text: str, retrieved_patterns: list[dict],
                 news_text: str = "", financials: dict = None) -> dict:
    prompt = _build_prompt(
        _THEME_ROLE, _THEME_GUIDE,
        current_text, format_retrieved_patterns(retrieved_patterns),
        news_text, financials or {}
    )
    raw = call_llm(_SYSTEM_BASE, prompt)
    result = parse_json_response(raw)
    if not result:
        logger.warning(f"theme_expert 파싱 실패: {raw[:100]}")
        return _default_signal()
    result["expert"] = "theme"
    return result


# ── 배당주 전문가 ────────────────────────────────────────────────────────────

_DIVIDEND_ROLE = "당신은 한국 배당주(고배당, 방어적 섹터, 안정 수익) 전문 애널리스트입니다."
_DIVIDEND_GUIDE = """다음 관점에서 분석하세요:
1. 배당 수익률 및 배당 지속 가능성
2. 방어적 특성 (시장 하락 시 상대 강도)
3. 실적 안정성 (분기별 이익 변동성)
4. 금리 환경과의 관계 (채권 대비 매력도)"""


def dividend_expert(current_text: str, retrieved_patterns: list[dict],
                    news_text: str = "", financials: dict = None) -> dict:
    prompt = _build_prompt(
        _DIVIDEND_ROLE, _DIVIDEND_GUIDE,
        current_text, format_retrieved_patterns(retrieved_patterns),
        news_text, financials or {}
    )
    raw = call_llm(_SYSTEM_BASE, prompt)
    result = parse_json_response(raw)
    if not result:
        logger.warning(f"dividend_expert 파싱 실패: {raw[:100]}")
        return _default_signal()
    result["expert"] = "dividend"
    return result


# ── 위기종목 전문가 ──────────────────────────────────────────────────────────

_CRISIS_ROLE = "당신은 한국 주식 위기종목(급락, 손절, 반등 가능성) 전문 애널리스트입니다."
_CRISIS_GUIDE = """다음 관점에서 분석하세요:
1. 급락 원인 분석 (일시적 vs 구조적)
2. 반등 가능성 및 지지선
3. 손절 vs 보유 판단 기준
4. 유사 급락 사례 회복 패턴
※ SELL 신호는 명확한 근거와 함께 제시"""


def crisis_expert(current_text: str, retrieved_patterns: list[dict],
                  news_text: str = "", financials: dict = None) -> dict:
    prompt = _build_prompt(
        _CRISIS_ROLE, _CRISIS_GUIDE,
        current_text, format_retrieved_patterns(retrieved_patterns),
        news_text, financials or {}
    )
    raw = call_llm(_SYSTEM_BASE, prompt)
    result = parse_json_response(raw)
    if not result:
        logger.warning(f"crisis_expert 파싱 실패: {raw[:100]}")
        return _default_signal()
    result["expert"] = "crisis"
    return result


# ── 라우터 ───────────────────────────────────────────────────────────────────

EXPERT_FUNCS = {
    "growth": growth_expert,
    "value": value_expert,
    "theme": theme_expert,
    "dividend": dividend_expert,
    "crisis": crisis_expert,
}


def run_experts(
    expert_names: list[str],
    current_text: str,
    retrieved_patterns: list[dict],
    news_text: str = "",
    financials: dict = None,
) -> list[dict]:
    """선택된 전문가 에이전트 실행 → 결과 목록 반환"""
    results = []
    for name in expert_names:
        fn = EXPERT_FUNCS.get(name)
        if fn is None:
            logger.warning(f"알 수 없는 전문가: {name}")
            continue
        res = fn(current_text, retrieved_patterns, news_text, financials)
        results.append(res)
    return results


def _default_signal() -> dict:
    return {
        "signal": "HOLD",
        "confidence": 0.5,
        "target_return": 0.0,
        "horizon_days": 5,
        "score": 5,
        "risks": ["분석 불가"],
        "reason": "LLM 응답 파싱 실패",
    }
