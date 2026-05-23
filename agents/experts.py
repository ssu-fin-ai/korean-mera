"""5개 전문가 에이전트 (성장주/가치주/테마주/배당주/위기종목) — LangGraph 노드

각 전문가는 스크리너가 선별한 Top-N 후보 종목을 배치로 받아 LLM 1회 호출로
최대 5개 추천 종목을 선택하고, 각 종목에 대해 근거·긍정·리스크를 출력한다.
"""

from agents.base import call_llm, parse_json_response
from loguru import logger

_SYSTEM_BASE = "당신은 한국 주식시장 전문 애널리스트입니다. 반드시 JSON만 응답하세요."

_JSON_FORMAT = """
출력 형식 (JSON만, 다른 텍스트 없이):
{
  "picks": [
    {
      "ticker": "005930",
      "name": "종목명",
      "signal": "BUY",
      "confidence": 0.85,
      "target_return": 0.10,
      "horizon_days": 10,
      "score": 8,
      "reason": "핵심 투자 근거 (2~3문장)",
      "pros": ["긍정 요인1", "긍정 요인2", "긍정 요인3"],
      "cons": ["리스크 요인1", "리스크 요인2"]
    }
  ]
}
signal: BUY | HOLD | SELL
confidence: 0.0 ~ 1.0
target_return: 예상 수익률 (소수, 예: 0.10 = +10%)
horizon_days: 투자 기간(일)
score: 매력도 1~10점
pros: 긍정 요인 2개 이상 (구체적 수치 포함)
cons: 리스크 1개 이상 (구체적 수치 포함)"""


# ── 포맷팅 헬퍼 ──────────────────────────────────────────────────────────────

def _fmt_won(v) -> str:
    try:
        n = int(str(v).replace(",", ""))
        if abs(n) >= 1_000_000_000_000:
            return f"{n / 1e12:.1f}조"
        if abs(n) >= 100_000_000:
            return f"{n / 1e8:.0f}억"
        return str(n)
    except Exception:
        return str(v)


def _pct(v, multiply: bool = False) -> str:
    if v is None:
        return "N/A"
    val = float(v) * (100 if multiply else 1)
    return f"{'+' if val >= 0 else ''}{val:.1f}%"


def _pattern_rets(retrieved: list) -> str:
    rets = []
    for p in retrieved[:3]:
        lbl = p.get("metadata", {}).get("label_5d")
        if lbl not in (None, "", "N/A"):
            try:
                rets.append(f"{float(lbl) * 100:+.1f}%")
            except Exception:
                pass
    return " | ".join(rets) if rets else ""


# ── 전문가별 후보 포맷터 ──────────────────────────────────────────────────────

def _fmt_growth(c: dict) -> str:
    f, s = c["financials"], c["snapshot"]
    price = f.get("current_price")
    mktcap = f.get("mktcap")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'} | 시총:{_fmt_won(mktcap) if mktcap else 'N/A'}",
        f"PER:{f.get('per', 'N/A')} | EPS:{f.get('eps', 'N/A')}원 | ROE:{f.get('roe', 'N/A')}%",
    ]
    yoy_parts = []
    if f.get("revenue_yoy") is not None:
        yoy_parts.append(f"매출YoY:{float(f['revenue_yoy']):+.1f}%")
    if f.get("op_income_yoy") is not None:
        yoy_parts.append(f"영업이익YoY:{float(f['op_income_yoy']):+.1f}%")
    if f.get("net_income_yoy") is not None:
        yoy_parts.append(f"순이익YoY:{float(f['net_income_yoy']):+.1f}%")
    if yoy_parts:
        lines.append(" | ".join(yoy_parts))
    lines.append(
        f"RSI:{s.get('rsi', 50):.0f} | MACD:{'+양전환' if s.get('macd_diff', 0) > 0 else '-음전환'} | "
        f"거래량비율:{s.get('volume_ratio', 1):.1f}x | 5일:{_pct(s.get('ret_5d'), multiply=True)}"
    )
    sr = f.get("short_ratio")
    if sr is not None:
        lines.append(f"공매도:{sr}% | 5일평균:{f.get('short_ratio_5d_avg', 'N/A')}%")
    pr = _pattern_rets(c.get("retrieved_patterns", []))
    if pr:
        lines.append(f"유사패턴수익률: {pr}")
    return "\n".join(lines)


def _fmt_value(c: dict) -> str:
    f = c["financials"]
    price = f.get("current_price")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'}",
        f"PER:{f.get('per', 'N/A')} | PBR:{f.get('pbr', 'N/A')} | "
        f"EPS:{f.get('eps', 'N/A')}원 | BPS:{f.get('bps', 'N/A')}원",
    ]
    pv_per = f.get("per_vs_sector")
    pv_pbr = f.get("pbr_vs_sector")
    if pv_per is not None or pv_pbr is not None:
        lines.append(
            f"섹터대비 PER:{_pct(pv_per, multiply=True)} | PBR:{_pct(pv_pbr, multiply=True)}"
        )
    gn = f.get("graham_number")
    if gn and price:
        lines.append(f"Graham Number:{int(gn):,}원 (현재가비율:{round(float(price) / float(gn), 2)}x)")
    lines.append(f"ROE:{f.get('roe', 'N/A')}% | ROA:{f.get('roa', 'N/A')}%")
    lines.append(
        f"부채비율:{f.get('debt_ratio', 'N/A')}% | 유동비율:{f.get('current_ratio', 'N/A')}% | "
        f"이자보상:{f.get('interest_coverage', 'N/A')}x"
    )
    lines.append(
        f"배당수익률:{f.get('div', 'N/A')}% | 배당성향:{f.get('payout_ratio', 'N/A')}% | "
        f"FCF:{_fmt_won(f['fcf']) if f.get('fcf') is not None else 'N/A'}"
    )
    h52 = f.get("pct_from_52w_high")
    l52 = f.get("pct_from_52w_low")
    if h52 is not None:
        lines.append(f"52주고점대비:{_pct(h52, multiply=True)} | 저점대비:{_pct(l52, multiply=True)}")
    return "\n".join(lines)


def _fmt_theme(c: dict) -> str:
    f, s = c["financials"], c["snapshot"]
    price = f.get("current_price")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'}",
        f"RSI:{s.get('rsi', 50):.0f} | MACD:{'+양전환' if s.get('macd_diff', 0) > 0 else '-음전환'} | "
        f"거래량비율:{s.get('volume_ratio', 1):.1f}x",
        f"5일:{_pct(s.get('ret_5d'), multiply=True)} | 20일:{_pct(s.get('ret_20d'), multiply=True)}",
    ]
    sr = f.get("short_ratio")
    if sr is not None:
        lines.append(f"공매도:{sr}% | 5일평균:{f.get('short_ratio_5d_avg', 'N/A')}%")
    h52 = f.get("pct_from_52w_high")
    if h52 is not None:
        lines.append(f"52주고점대비:{_pct(h52, multiply=True)}")
    if c.get("news_text"):
        lines.append(f"최근공시: {c['news_text'][:150]}")
    pr = _pattern_rets(c.get("retrieved_patterns", []))
    if pr:
        lines.append(f"유사패턴수익률: {pr}")
    return "\n".join(lines)


def _fmt_dividend(c: dict) -> str:
    f, s = c["financials"], c["snapshot"]
    price = f.get("current_price")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'}",
        f"배당수익률:{f.get('div', 'N/A')}% | DPS:{f.get('dps', 'N/A')}원 | "
        f"배당성향:{f.get('payout_ratio', 'N/A')}%",
        f"FCF:{_fmt_won(f['fcf']) if f.get('fcf') is not None else 'N/A'} | "
        f"OCF:{_fmt_won(f['ocf']) if f.get('ocf') is not None else 'N/A'}",
        f"ROE:{f.get('roe', 'N/A')}% | ROA:{f.get('roa', 'N/A')}%",
        f"부채비율:{f.get('debt_ratio', 'N/A')}% | 이자보상:{f.get('interest_coverage', 'N/A')}x",
        f"베타:{round(float(s.get('beta_20d') or 1), 2)} | "
        f"역사적변동성:{round(float(s.get('hist_vol_20') or 0) * 100, 1)}%",
        f"5일:{_pct(s.get('ret_5d'), multiply=True)} | 20일:{_pct(s.get('ret_20d'), multiply=True)}",
    ]
    return "\n".join(lines)


def _fmt_crisis(c: dict) -> str:
    f, s = c["financials"], c["snapshot"]
    price = f.get("current_price")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'}",
        f"5일:{_pct(s.get('ret_5d'), multiply=True)} | "
        f"20일:{_pct(s.get('ret_20d'), multiply=True)} | "
        f"1일:{_pct(s.get('ret_1d'), multiply=True)}",
        f"RSI:{s.get('rsi', 50):.0f} | BB위치:{s.get('bb_pct', 0.5):.2f} | "
        f"거래량비율:{s.get('volume_ratio', 1):.1f}x",
    ]
    h52 = f.get("pct_from_52w_high")
    l52 = f.get("pct_from_52w_low")
    if h52 is not None:
        lines.append(f"52주고점대비:{_pct(h52, multiply=True)} | 저점대비:{_pct(l52, multiply=True)}")
    sr = f.get("short_ratio")
    if sr is not None:
        lines.append(f"공매도:{sr}% | 5일평균:{f.get('short_ratio_5d_avg', 'N/A')}%")
    lines.append(
        f"부채비율:{f.get('debt_ratio', 'N/A')}% | 유동비율:{f.get('current_ratio', 'N/A')}% | "
        f"이자보상:{f.get('interest_coverage', 'N/A')}x"
    )
    if c.get("news_text"):
        lines.append(f"최근공시: {c['news_text'][:150]}")
    pr = _pattern_rets(c.get("retrieved_patterns", []))
    if pr:
        lines.append(f"유사패턴수익률: {pr}")
    return "\n".join(lines)


_FORMATTERS = {
    "growth": _fmt_growth,
    "value": _fmt_value,
    "theme": _fmt_theme,
    "dividend": _fmt_dividend,
    "crisis": _fmt_crisis,
}


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────

def _build_prompt(role_desc: str, guide: str, candidates: list[dict], expert: str) -> str:
    fmt = _FORMATTERS[expert]
    blocks = "\n\n".join(f"{i}. {fmt(c)}" for i, c in enumerate(candidates, 1))
    return f"""{role_desc}

아래 {len(candidates)}개 후보 종목에서 투자 매력도가 높은 종목을 선별하여 상세 분석하세요.
BUY 종목을 우선하되, HOLD/SELL도 근거가 명확하면 포함하세요.
각 종목의 pros·cons에는 반드시 구체적인 수치를 포함하세요.

## 후보 종목

{blocks}

## 분석 요청
{guide}

{_JSON_FORMAT}"""


# ── 전문가 노드 ───────────────────────────────────────────────────────────────

_GROWTH_ROLE = "당신은 한국 성장주(고PER, 실적 모멘텀) 전문 애널리스트입니다."
_GROWTH_GUIDE = """다음 관점에서 최대 5개 종목을 선택하세요:
1. 실적 모멘텀 — 매출·영업이익 YoY 성장률이 높은 순
2. 주가 모멘텀 — RSI 45~70, MACD 양전환, 거래량 동반 상승
3. 섹터 성장성 — AI·반도체·2차전지·바이오 등 고성장 섹터 우선
4. 유사 패턴의 과거 5일 수익률"""


def growth_node(state: dict) -> dict:
    candidates = state.get("growth_candidates", [])
    if not candidates:
        return {"growth_picks": []}
    prompt = _build_prompt(_GROWTH_ROLE, _GROWTH_GUIDE, candidates, "growth")
    raw = call_llm(_SYSTEM_BASE, prompt)
    picks = _parse_picks(raw, "growth")
    logger.info(f"성장주 전문가: {len(picks)}개 선택")
    return {"growth_picks": picks}


_VALUE_ROLE = "당신은 한국 가치주(저PBR/PER, 자산가치, 안정 현금흐름) 전문 애널리스트입니다."
_VALUE_GUIDE = """다음 관점에서 최대 5개 종목을 선택하세요:
1. 밸류에이션 — PBR < 1.5, 섹터 대비 저PER
2. Graham Number 대비 현재가 위치 (낮을수록 매력적)
3. 재무 안정성 — 부채비율·이자보상배율
4. 저평가 해소 촉매 — 배당수익률·FCF"""


def value_node(state: dict) -> dict:
    candidates = state.get("value_candidates", [])
    if not candidates:
        return {"value_picks": []}
    prompt = _build_prompt(_VALUE_ROLE, _VALUE_GUIDE, candidates, "value")
    raw = call_llm(_SYSTEM_BASE, prompt)
    picks = _parse_picks(raw, "value")
    logger.info(f"가치주 전문가: {len(picks)}개 선택")
    return {"value_picks": picks}


_THEME_ROLE = "당신은 한국 테마주(정책수혜, 이슈 모멘텀, 단기 급등) 전문 애널리스트입니다."
_THEME_GUIDE = """다음 관점에서 최대 5개 종목을 선택하세요:
1. 최근 공시·정책 수혜 직접성 (공시 내용 반영)
2. 거래량 폭발 + 주가 모멘텀 (RSI 70 미만 우선)
3. 테마 사이클 위치 (초기 진입 우선, 말기 배제)
4. 유사 급등 이후 과거 수익률 패턴"""


def theme_node(state: dict) -> dict:
    candidates = state.get("theme_candidates", [])
    if not candidates:
        return {"theme_picks": []}
    prompt = _build_prompt(_THEME_ROLE, _THEME_GUIDE, candidates, "theme")
    raw = call_llm(_SYSTEM_BASE, prompt)
    picks = _parse_picks(raw, "theme")
    logger.info(f"테마주 전문가: {len(picks)}개 선택")
    return {"theme_picks": picks}


_DIVIDEND_ROLE = "당신은 한국 배당주(고배당, 방어적 섹터, 안정 수익) 전문 애널리스트입니다."
_DIVIDEND_GUIDE = """다음 관점에서 최대 5개 종목을 선택하세요:
1. 배당수익률 + FCF 기반 지속 가능성 (배당성향 30~70% 적정)
2. 방어적 특성 — 낮은 베타·변동성
3. 재무 건전성 — 부채비율·이자보상배율
4. 금리 환경 대비 배당 매력도"""


def dividend_node(state: dict) -> dict:
    candidates = state.get("dividend_candidates", [])
    if not candidates:
        return {"dividend_picks": []}
    prompt = _build_prompt(_DIVIDEND_ROLE, _DIVIDEND_GUIDE, candidates, "dividend")
    raw = call_llm(_SYSTEM_BASE, prompt)
    picks = _parse_picks(raw, "dividend")
    logger.info(f"배당주 전문가: {len(picks)}개 선택")
    return {"dividend_picks": picks}


_CRISIS_ROLE = "당신은 한국 주식 위기종목(급락 후 반등 가능성) 전문 애널리스트입니다."
_CRISIS_GUIDE = """다음 관점에서 반등 가능성이 높은 종목을 최대 5개 선택하세요:
1. 급락 원인 분석 — 일시적 악재 vs 구조적 문제 구분
2. 기술적 과매도 정도 — RSI 30 이하, BB 하단 접근
3. 재무 건전성 — 이자보상·유동비율 (반등 지속 가능성)
4. 유사 급락 이후 역사적 회복 패턴"""


def crisis_node(state: dict) -> dict:
    candidates = state.get("crisis_candidates", [])
    if not candidates:
        return {"crisis_picks": []}
    prompt = _build_prompt(_CRISIS_ROLE, _CRISIS_GUIDE, candidates, "crisis")
    raw = call_llm(_SYSTEM_BASE, prompt)
    picks = _parse_picks(raw, "crisis")
    logger.info(f"위기종목 전문가: {len(picks)}개 선택")
    return {"crisis_picks": picks}


# ── 파싱 헬퍼 ────────────────────────────────────────────────────────────────

def _parse_picks(raw: str, expert: str) -> list[dict]:
    result = parse_json_response(raw)
    if not result:
        logger.warning(f"{expert} 파싱 실패: {raw[:200]}")
        return []
    picks = result.get("picks", [])
    if not isinstance(picks, list):
        logger.warning(f"{expert} picks 필드가 리스트 아님")
        return []
    for p in picks:
        p["expert"] = expert
        p.setdefault("pros", [])
        p.setdefault("cons", [])
        p.setdefault("reason", "")
        try:
            p["confidence"] = float(p.get("confidence", 0.5))
        except Exception:
            p["confidence"] = 0.5
        try:
            p["score"] = int(p.get("score", 5))
        except Exception:
            p["score"] = 5
        try:
            p["target_return"] = float(p.get("target_return", 0.0))
        except Exception:
            p["target_return"] = 0.0
        try:
            p["horizon_days"] = int(p.get("horizon_days", 10))
        except Exception:
            p["horizon_days"] = 10
    return picks
