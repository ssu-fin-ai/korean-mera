"""주식 패턴을 LLM 임베딩용 텍스트로 변환

ChromaDB에 저장할 패턴 텍스트와 DART 공시 텍스트를 생성한다.
임베딩 모델(text-embedding-3-small)이 의미적 유사도를 잘 계산할 수 있도록
숫자보다는 자연어 레이블 중심으로 구성한다.
"""

import pandas as pd


def _fmt(v: float, pct: bool = False) -> str:
    """소수값을 포맷팅 (pct=True면 ×100하여 % 표시)"""
    if pct:
        return f"{v*100:+.1f}%"
    return f"{v:.2f}"


def _rsi_label(rsi: float) -> str:
    """RSI 수치 → 한국어 레이블 (과매수/과매도/강세/약세/중립)"""
    if rsi >= 70:
        return "과매수(70+)"
    if rsi <= 30:
        return "과매도(30-)"
    if rsi >= 60:
        return "강세"
    if rsi <= 40:
        return "약세"
    return "중립"


def _bb_label(bb_pct: float) -> str:
    """볼린저밴드 위치(0~1) → 한국어 레이블 (상단돌파/하단접근/상단대/하단대/중간대)"""
    if bb_pct >= 0.9:
        return "상단돌파"
    if bb_pct <= 0.1:
        return "하단접근"
    if bb_pct >= 0.7:
        return "상단대"
    if bb_pct <= 0.3:
        return "하단대"
    return "중간대"


def _vol_label(vol_ratio: float) -> str:
    """거래량비율(20일 평균 대비) → 한국어 레이블"""
    if vol_ratio >= 3.0:
        return "폭발적급증(3배+)"
    if vol_ratio >= 2.0:
        return "급증(2배+)"
    if vol_ratio >= 1.5:
        return "증가"
    if vol_ratio <= 0.5:
        return "급감"
    return "정상"


def _macd_label(macd_diff: float) -> str:
    """MACD 히스토그램(MACD - Signal) → 추세 레이블
    임계값 0.002는 정규화된 값 기준으로 noise 필터링용
    """
    if macd_diff > 0.002:
        return "골든크로스/상승세"
    if macd_diff < -0.002:
        return "데드크로스/하락세"
    return "중립"


def _trend_label(close_to_ma20: float, close_to_ma60: float) -> str:
    """MA20/MA60 대비 현재가 위치 → 추세 레이블
    5% 기준으로 강한 추세 vs 단기 방향 구분
    """
    if close_to_ma20 > 0.05 and close_to_ma60 > 0.05:
        return "강한상승추세"
    if close_to_ma20 < -0.05 and close_to_ma60 < -0.05:
        return "강한하락추세"
    if close_to_ma20 > 0:
        return "단기상승"
    return "단기하락"


def generate_pattern_text(
    ticker: str,
    name: str,
    sector: str,
    market: str,
    snapshot: dict,
    label_5d: float | None = None,
    filings: list[dict] | None = None,
) -> str:
    """패턴 스냅샷 dict → ChromaDB 저장용 임베딩 텍스트 생성

    숫자 그대로보다 자연어 레이블을 함께 써서
    임베딩 모델이 "과매도+하락추세" 같은 패턴을 의미적으로 유사하게 처리하게 한다.

    label_5d가 있으면 "이후5일수익률" 줄을 추가 → 과거 DB 저장용
    label_5d가 없으면 해당 줄 생략 → 현재 종목 쿼리용
    """
    ret_5d = snapshot.get("ret_5d", 0)
    ret_20d = snapshot.get("ret_20d", 0)
    rsi = snapshot.get("rsi", 50)
    macd_diff = snapshot.get("macd_diff", 0)
    bb_pct = snapshot.get("bb_pct", 0.5)
    vol_ratio = snapshot.get("volume_ratio", 1.0)
    hist_vol = snapshot.get("hist_vol_20", 0.3)
    close_to_ma20 = snapshot.get("close_to_ma20", 0)
    close_to_ma60 = snapshot.get("close_to_ma60", 0)
    adx = snapshot.get("adx", 20)   # ADX: 추세 강도 (25+ = 뚜렷한 추세)
    mfi = snapshot.get("mfi", 50)   # MFI: 자금 흐름 지수 (거래량 가중 RSI)

    trend = _trend_label(close_to_ma20, close_to_ma60)

    # 과거 패턴 저장 시에만 실제 수익률 줄 추가
    result_line = ""
    if label_5d is not None:
        result_line = f"이후5일수익률: {_fmt(label_5d, pct=True)} | "

    # DART 공시 제목 (최대 3개)을 텍스트에 포함해 임베딩 품질 향상
    filing_line = ""
    if filings:
        titles = [f["report_nm"] for f in filings[:3]]
        filing_line = f"\n최근공시: {', '.join(titles)}"

    text = (
        f"종목: {ticker} {name} | 시장: {market} | 섹터: {sector}\n"
        f"수익률: 5일{_fmt(ret_5d, pct=True)} 20일{_fmt(ret_20d, pct=True)}\n"
        f"추세: {trend} | MA20대비{_fmt(close_to_ma20*100, pct=False)}% "
        f"MA60대비{_fmt(close_to_ma60*100, pct=False)}%\n"
        f"RSI: {rsi:.0f}({_rsi_label(rsi)}) | MACD: {_macd_label(macd_diff)}\n"
        f"볼린저: {_bb_label(bb_pct)}({bb_pct:.2f}) | 거래량: {_vol_label(vol_ratio)}\n"
        f"역사적변동성(20일): {hist_vol*100:.1f}% | ADX: {adx:.0f} | MFI: {mfi:.0f}\n"
        f"{result_line}"
        f"{filing_line}"
    ).strip()

    return text


def generate_news_text(ticker: str, name: str, items: list[dict]) -> str:
    """DART 공시 목록 → 전문가 프롬프트용 텍스트 (최대 5개)

    items는 get_recent_filings()가 반환한 dict 리스트.
    각 항목에서 접수일자(rcept_dt)와 보고서명(report_nm)을 추출한다.
    """
    if not items:
        return f"{ticker} {name}: 최근 공시/뉴스 없음"
    lines = [f"{it.get('rcept_dt','')}: {it.get('report_nm','')}" for it in items[:5]]
    return f"{ticker} {name} 최근공시\n" + "\n".join(lines)
