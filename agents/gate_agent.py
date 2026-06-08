"""GateNet: 종목 패턴별 전문가 가중치 산출

MERA 논문의 GateNet을 LLM 기반으로 구현한다.
종목의 현재 기술적 패턴과 유사 과거 사례를 분석해
5개 전문가(성장주/가치주/테마주/배당주/위기종목) 각각의 적합도를 0~1 점수로 산출한다.

역할: aggregator에서 각 전문가 picks에 이 가중치를 곱해
"이 종목 패턴에 맞는 전문가의 의견에 더 높은 비중"을 부여한다.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger

from agents.base import call_llm, format_retrieved_patterns, parse_json_response

# 5개 전문가 이름 (가중치 dict의 키로 사용)
_EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]

# 파싱 실패 시 모든 전문가에게 동일 가중치 1.0 부여 (neutral fallback)
_DEFAULT_WEIGHTS = {e: 1.0 for e in _EXPERTS}

SYSTEM = """당신은 주식 패턴 분류 전문가입니다.
현재 종목의 패턴과 유사한 과거 사례를 분석하여
각 투자 전문가의 적합도를 평가합니다.
반드시 JSON 형식으로만 응답하세요."""

# 현재 종목 상태와 RAG 유사 패턴을 함께 보여주어
# LLM이 "이 종목은 어느 전문가가 잘 맞는지" 판단하게 하는 프롬프트
PROMPT_TEMPLATE = """## 현재 종목 상태
{current_text}

## 유사한 과거 사례 (Top-{k})
{retrieved_text}

## 전문가 유형
- growth   : 성장주 전문가 — 실적 모멘텀, 거래량 급증, 고성장
- value    : 가치주 전문가 — 저PBR/PER, 안정 현금흐름, 자산가치
- theme    : 테마주 전문가 — 정책수혜, 이슈 급등, 단기 모멘텀
- dividend : 배당주 전문가 — 고배당, 방어적 섹터, 저변동성
- crisis   : 위기종목 전문가 — 급락, 과매도, 반등 가능성

## 지시
현재 패턴에 대한 각 전문가의 적합도를 0.0~1.0으로 평가하세요.
- 1.0: 이 패턴의 핵심 전문가
- 0.5: 참고 가능
- 0.0: 이 패턴과 무관

출력 형식:
{{
  "weights": {{
    "growth": 0.8,
    "value": 0.1,
    "theme": 0.3,
    "dividend": 0.0,
    "crisis": 0.2
  }},
  "pattern_type": "실적모멘텀 성장주",
  "reason": "RSI 68, 거래량 2.3배로 성장 모멘텀 강함"
}}"""


def weight(current_text: str, retrieved_patterns: list[dict]) -> dict:
    """종목의 현재 패턴 텍스트 + RAG 패턴 → 전문가별 가중치 dict 반환

    반환 형식:
      {
        "weights": {"growth": 0.8, "value": 0.1, ...},
        "pattern_type": "실적모멘텀 성장주",
        "reason": "..."
      }
    """
    # RAG 패턴을 사람이 읽을 수 있는 텍스트로 변환
    retrieved_text = format_retrieved_patterns(retrieved_patterns)
    prompt = PROMPT_TEMPLATE.format(
        current_text=current_text,
        retrieved_text=retrieved_text,
        k=len(retrieved_patterns),
    )
    raw = call_llm(SYSTEM, prompt)
    result = parse_json_response(raw)

    if not result or "weights" not in result:
        # 파싱 실패 시 기본값 사용 (모든 전문가 동일 가중치)
        logger.warning(f"GateNet 파싱 실패, 기본값 사용: {raw[:80]}")
        return {"weights": _DEFAULT_WEIGHTS, "pattern_type": "미분류", "reason": "파싱 실패"}

    # LLM이 범위를 벗어난 값을 반환할 수 있으므로 0.0~1.0으로 클램핑
    w = result["weights"]
    result["weights"] = {e: max(0.0, min(1.0, float(w.get(e, 1.0)))) for e in _EXPERTS}
    return result


def gate_node(state: dict) -> dict:
    """스크리너 후보 종목별 전문가 가중치 계산 (ThreadPoolExecutor 병렬 처리)

    각 전문가의 candidates에서 고유 종목을 추출한 뒤
    각 종목마다 GateNet LLM을 병렬 호출해 가중치를 계산한다.
    결과는 {ticker: {expert: weight}} 형태의 gate_weights_map으로 반환된다.
    """
    # 모든 전문가 candidates에서 중복 없이 고유 종목 수집
    unique: dict[str, dict] = {}
    for expert in _EXPERTS:
        for c in state.get(f"{expert}_candidates", []):
            if c["ticker"] not in unique:
                unique[c["ticker"]] = c

    gate_weights_map: dict[str, dict] = {}

    def _run(ticker: str, c: dict):
        """단일 종목 GateNet 호출 (스레드풀 내 실행)"""
        try:
            result = weight(c["current_text"], c["retrieved_patterns"])
            return ticker, result["weights"]
        except Exception as e:
            logger.warning(f"GateNet {ticker} 실패: {e}")
            return ticker, _DEFAULT_WEIGHTS.copy()

    # 최대 5개 스레드로 종목별 LLM 호출 병렬화
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(_run, t, c): t for t, c in unique.items()}
        for f in as_completed(futures):
            ticker, w = f.result()
            gate_weights_map[ticker] = w

    logger.info(f"GateNet 완료: {len(gate_weights_map)}종목 가중치 산출")
    return {"gate_weights_map": gate_weights_map}
