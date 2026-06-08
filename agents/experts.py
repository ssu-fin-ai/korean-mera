"""5개 전문가 에이전트 (성장주/가치주/테마주/배당주/위기종목) — LangGraph 노드

각 전문가는 스크리너가 선별한 Top-N 후보 종목을 배치로 받아 LLM 1회 호출로
최대 5개 추천 종목을 선택하고, 각 종목에 대해 근거·긍정·리스크를 출력한다.

공통 처리 흐름:
  1. 후보 종목 리스트를 전문가 특화 포맷터로 텍스트 변환
  2. 전문가 역할·분석 지침 + JSON 출력 형식을 합쳐 프롬프트 구성
  3. Claude LLM 호출
  4. JSON 파싱 → picks 리스트 정규화 (타입 보정, 기본값 보완)
  5. LangGraph state에 {expert}_picks 키로 반환
"""

from agents.base import call_llm, parse_json_response
from loguru import logger

# 모든 전문가 공통 시스템 프롬프트: JSON 전용 응답 강제
_SYSTEM_BASE = "당신은 한국 주식시장 전문 애널리스트입니다. 반드시 JSON만 응답하세요."

# LLM이 반환해야 할 JSON 구조 명세 (프롬프트 말미에 항상 첨부)
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
    """숫자를 한국식 원 단위(조/억)로 변환 (예: 1500000000000 → '1.5조')"""
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
    """소수를 퍼센트 문자열로 변환 (multiply=True면 ×100 처리)"""
    if v is None:
        return "N/A"
    val = float(v) * (100 if multiply else 1)
    return f"{'+' if val >= 0 else ''}{val:.1f}%"


# label_key → 기간(일) 매핑 (RAG 섹션 텍스트 생성용)
_LABEL_DAYS = {"label_5d": 5, "label_10d": 10, "label_20d": 20}


def _fmt_rag_section(retrieved: list, label_key: str = "label_5d") -> str:
    """RAG 유사 패턴 수익률 통계 섹션 텍스트 생성

    상위 5개 패턴에서 유효한 수익률만 추출해
    유사도·수익률 목록과 적중률·평균수익 요약을 반환한다.
    LLM이 confidence를 조정할 근거 데이터로 활용.
    """
    days = _LABEL_DAYS.get(label_key, 5)
    valid = []
    for p in retrieved[:5]:
        lbl = p.get("metadata", {}).get(label_key)
        dist = p.get("distance", 1.0)
        if lbl not in (None, "", "N/A"):
            try:
                # (유사도%, 수익률%) 쌍으로 변환
                valid.append((round((1 - float(dist)) * 100), float(lbl) * 100))
            except Exception:
                pass
    if not valid:
        return ""
    lines = [f"[유사 과거 패턴 (RAG/{days}일 수익률)]"]
    for i, (sim, ret) in enumerate(valid, 1):
        lines.append(f"  {i}. 유사도:{sim}% → {days}일후:{ret:+.1f}%")
    rets = [r for _, r in valid]
    hit = sum(1 for r in rets if r > 0)  # 양수 수익률 = 적중
    avg = sum(rets) / len(rets)
    lines.append(f"  적중률:{hit}/{len(rets)} | 평균수익:{avg:+.1f}%")
    return "\n".join(lines)


# ── 전문가별 후보 포맷터 ──────────────────────────────────────────────────────
# 각 전문가가 중점을 두는 지표만 골라 텍스트로 변환한다.
# LLM 프롬프트에 포함되므로 너무 길면 토큰 낭비, 너무 짧으면 분석 품질 저하.

def _fmt_growth(c: dict) -> str:
    """성장주 포맷: 실적 YoY·기술적 모멘텀·공매도·RAG 패턴 중심"""
    f, s = c["financials"], c["snapshot"]
    price = f.get("current_price")
    mktcap = f.get("mktcap")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'} | 시총:{_fmt_won(mktcap) if mktcap else 'N/A'}",
        f"PER:{f.get('per', 'N/A')} | EPS:{f.get('eps', 'N/A')}원 | ROE:{f.get('roe', 'N/A')}%",
    ]
    # 매출·영업이익·순이익 YoY 성장률 (있는 것만 표시)
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
    if c.get("news_text"):
        lines.append(f"최근공시:\n{c['news_text']}")
    # RAG 유사 패턴 수익률 통계 (label_key는 monthly/weekly 모드에 따라 결정)
    rag = _fmt_rag_section(c.get("retrieved_patterns", []), label_key=c.get("rag_label_key", "label_20d"))
    if rag:
        lines.append(rag)
    return "\n".join(lines)


def _fmt_value(c: dict) -> str:
    """가치주 포맷: PBR/PER·Graham Number·재무 안정성·배당 중심"""
    f = c["financials"]
    price = f.get("current_price")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'}",
        f"PER:{f.get('per', 'N/A')} | PBR:{f.get('pbr', 'N/A')} | "
        f"EPS:{f.get('eps', 'N/A')}원 | BPS:{f.get('bps', 'N/A')}원",
    ]
    # 섹터 대비 PER/PBR 위치 (음수 = 섹터 평균보다 저평가)
    pv_per = f.get("per_vs_sector")
    pv_pbr = f.get("pbr_vs_sector")
    if pv_per is not None or pv_pbr is not None:
        lines.append(
            f"섹터대비 PER:{_pct(pv_per, multiply=True)} | PBR:{_pct(pv_pbr, multiply=True)}"
        )
    # Graham Number = √(22.5 × EPS × BPS) — 벤저민 그레이엄의 내재가치 기준
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
    # 52주 고점/저점 대비 위치 (가치주 저평가 확인에 유용)
    h52 = f.get("pct_from_52w_high")
    l52 = f.get("pct_from_52w_low")
    if h52 is not None:
        lines.append(f"52주고점대비:{_pct(h52, multiply=True)} | 저점대비:{_pct(l52, multiply=True)}")
    if c.get("news_text"):
        lines.append(f"최근공시:\n{c['news_text']}")
    rag = _fmt_rag_section(c.get("retrieved_patterns", []), label_key=c.get("rag_label_key", "label_20d"))
    if rag:
        lines.append(rag)
    return "\n".join(lines)


def _fmt_theme(c: dict) -> str:
    """테마주 포맷: 거래량 폭발·단기 모멘텀·공매도·52주 위치 중심"""
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
        lines.append(f"최근공시:\n{c['news_text']}")
    rag = _fmt_rag_section(c.get("retrieved_patterns", []), label_key=c.get("rag_label_key", "label_20d"))
    if rag:
        lines.append(rag)
    return "\n".join(lines)


def _fmt_dividend(c: dict) -> str:
    """배당주 포맷: 배당수익률·FCF·베타·변동성 중심 (방어적 특성 강조)"""
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
        # 베타 1 미만 = 시장보다 변동성 낮음 (방어적 배당주에 적합)
        f"베타:{round(float(s.get('beta_20d') or 1), 2)} | "
        f"역사적변동성:{round(float(s.get('hist_vol_20') or 0) * 100, 1)}%",
        f"5일:{_pct(s.get('ret_5d'), multiply=True)} | 20일:{_pct(s.get('ret_20d'), multiply=True)}",
    ]
    if c.get("news_text"):
        lines.append(f"최근공시:\n{c['news_text']}")
    rag = _fmt_rag_section(c.get("retrieved_patterns", []), label_key=c.get("rag_label_key", "label_20d"))
    if rag:
        lines.append(rag)
    return "\n".join(lines)


def _fmt_crisis(c: dict) -> str:
    """위기종목 포맷: 급락 폭·과매도 지표·재무 건전성 중심 (반등 가능성 판단)"""
    f, s = c["financials"], c["snapshot"]
    price = f.get("current_price")
    lines = [
        f"### {c['name']} ({c['ticker']}) — {c['sector']}",
        f"현재가:{f'{price:,}원' if price else 'N/A'}",
        # 단기 수익률 (음수 = 급락)
        f"5일:{_pct(s.get('ret_5d'), multiply=True)} | "
        f"20일:{_pct(s.get('ret_20d'), multiply=True)} | "
        f"1일:{_pct(s.get('ret_1d'), multiply=True)}",
        # RSI 30 이하 = 과매도, BB 하단 = 지지선 접근
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
    # 재무 건전성: 부채비율·유동비율·이자보상배율로 반등 지속 가능성 확인
    lines.append(
        f"부채비율:{f.get('debt_ratio', 'N/A')}% | 유동비율:{f.get('current_ratio', 'N/A')}% | "
        f"이자보상:{f.get('interest_coverage', 'N/A')}x"
    )
    if c.get("news_text"):
        lines.append(f"최근공시:\n{c['news_text']}")
    rag = _fmt_rag_section(c.get("retrieved_patterns", []), label_key=c.get("rag_label_key", "label_20d"))
    if rag:
        lines.append(rag)
    return "\n".join(lines)


# 전문가 이름 → 포맷터 함수 매핑 (프롬프트 빌더에서 동적 선택)
_FORMATTERS = {
    "growth": _fmt_growth,
    "value": _fmt_value,
    "theme": _fmt_theme,
    "dividend": _fmt_dividend,
    "crisis": _fmt_crisis,
}


# ── 프롬프트 빌더 ─────────────────────────────────────────────────────────────

def _build_prompt(role_desc: str, guide: str, candidates: list[dict], expert: str) -> str:
    """전문가 역할·분석 지침·후보 종목 블록·JSON 형식을 합쳐 최종 프롬프트 생성

    각 후보 종목은 expert에 맞는 포맷터로 텍스트화되어 번호와 함께 나열된다.
    RAG 활용 지침을 포함해 LLM이 과거 패턴 통계를 confidence 조정에 반영하도록 유도한다.
    """
    fmt = _FORMATTERS[expert]
    # 후보 종목 각각을 포맷팅해 번호 붙여 합침
    blocks = "\n\n".join(f"{i}. {fmt(c)}" for i, c in enumerate(candidates, 1))
    return f"""{role_desc}

아래 {len(candidates)}개 후보 종목에서 투자 매력도가 높은 종목을 선별하여 상세 분석하세요.
BUY 종목을 우선하되, HOLD/SELL도 근거가 명확하면 포함하세요.
각 종목의 pros·cons에는 반드시 구체적인 수치를 포함하세요.

[RAG 활용 지침]
각 종목의 [유사 과거 패턴 (RAG)] 섹션을 반드시 참고하세요.
- 적중률이 높고(4/5 이상) 평균수익이 플러스인 종목은 confidence를 높게 설정하세요.
- 적중률이 낮거나(2/5 이하) 평균수익이 마이너스인 종목은 confidence를 낮추거나 HOLD로 판단하세요.
- RAG 통계는 펀더멘털 분석을 보완하는 증거로 활용하세요. (RAG만으로 BUY하지 말것)

## 후보 종목

{blocks}

## 분석 요청
{guide}

{_JSON_FORMAT}"""


# ── 전문가 노드 ───────────────────────────────────────────────────────────────
# 각 노드는 LangGraph state에서 자신의 candidates를 꺼내 LLM 분석 후 picks를 반환한다.

_GROWTH_ROLE = "당신은 한국 성장주(고PER, 실적 모멘텀) 전문 애널리스트입니다."
_GROWTH_GUIDE = """다음 관점에서 최대 5개 종목을 선택하세요:
1. 실적 모멘텀 — 매출·영업이익 YoY 성장률이 높은 순
2. 주가 모멘텀 — RSI 45~70, MACD 양전환, 거래량 동반 상승
3. 섹터 성장성 — AI·반도체·2차전지·바이오 등 고성장 섹터 우선
4. RAG 과거 패턴 — 유사 모멘텀 패턴의 적중률·평균수익을 confidence에 반영
5. 공시 활용:
   - 유상증자 → 희석 리스크로 confidence 하향
   - 대규모 수주·계약 체결 → 실적 모멘텀 강화로 BUY 가산점
   - 전환사채 발행 → 성장 재원 확보 vs 희석 리스크 균형 판단"""


def growth_node(state: dict) -> dict:
    """성장주 전문가 노드: growth_candidates → growth_picks"""
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
4. 저평가 해소 촉매 — 배당수익률·FCF
5. RAG 과거 패턴 — 유사 저평가 패턴에서 실제 주가 회복 여부를 confidence에 반영
6. 공시 활용:
   - 자기주식 취득 → 저평가 인정 + 주주환원 신호로 BUY 가산점
   - 전환사채 발행 → BPS·PBR 희석 우려로 confidence 하향
   - 유상증자 → 자산가치 희석, 저평가 해소 지연 리스크"""


def value_node(state: dict) -> dict:
    """가치주 전문가 노드: value_candidates → value_picks"""
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
4. RAG 과거 패턴 — 유사 급등 패턴 이후 실제 수익률 추이를 confidence에 반영
5. 공시 활용:
   - 주요사항보고서(수주·계약·MOU) → 테마 직접 수혜 확인으로 BUY 가산점
   - 조회공시 → 급등 원인 공식 확인, 테마 지속성 판단
   - 불성실공시 → 신뢰도 하락, confidence 대폭 하향"""


def theme_node(state: dict) -> dict:
    """테마주 전문가 노드: theme_candidates → theme_picks"""
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
4. 금리 환경 대비 배당 매력도
5. RAG 과거 패턴 — 유사 방어주 패턴의 안정성(손실 최소화)을 confidence에 반영
6. 공시 활용:
   - 배당 결정·증액 공시 → 배당 지속성 확인으로 BUY 가산점
   - 유상증자 → 배당 재원 희석 우려로 confidence 하향
   - 최대주주 변경 → 배당 정책 변화 가능성, 불확실성으로 HOLD 검토"""


def dividend_node(state: dict) -> dict:
    """배당주 전문가 노드: dividend_candidates → dividend_picks"""
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
4. RAG 과거 패턴 — 유사 급락 패턴 이후 실제 반등 성공률을 confidence에 반영 (적중률 높을수록 BUY 우선)
5. 공시 활용:
   - 횡령·배임·불성실공시 → 구조적 문제로 판단, SELL 또는 confidence 대폭 하향
   - 조회공시(풍문·보도 해명) → 악재 공식화 여부 확인, 일시적이면 반등 가능
   - 유상증자(긴급) → 재무 위기 신호, 반등 가능성 낮음으로 HOLD/SELL"""


def crisis_node(state: dict) -> dict:
    """위기종목 전문가 노드: crisis_candidates → crisis_picks"""
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
    """LLM 응답 JSON 파싱 + 타입 정규화 + 기본값 보완

    LLM이 반환한 picks 각 항목에 대해:
    - expert 필드 주입 (어떤 전문가가 선택했는지 추적)
    - pros/cons/reason 기본값 설정
    - confidence/score/target_return/horizon_days 안전한 타입 변환
    """
    result = parse_json_response(raw)
    if not result:
        logger.warning(f"{expert} 파싱 실패: {raw[:200]}")
        return []
    picks = result.get("picks", [])
    if not isinstance(picks, list):
        logger.warning(f"{expert} picks 필드가 리스트 아님")
        return []
    for p in picks:
        p["expert"] = expert  # 어느 전문가 에이전트의 결과인지 태그
        p.setdefault("pros", [])
        p.setdefault("cons", [])
        p.setdefault("reason", "")
        # LLM이 문자열로 반환할 수 있으므로 float/int 변환 (실패 시 기본값)
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
