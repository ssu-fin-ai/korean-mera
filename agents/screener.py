"""사전 스크리닝 노드: 전체 종목 데이터 수집 + 전문가별 후보 필터링

전체 유니버스(KOSPI200+KOSDAQ150)에서 각 전문가 유형에 맞는 상위 N개 종목을 선별한다.
LLM 호출 없이 휴리스틱 스코어만으로 빠르게 필터링해 후속 LLM 비용을 줄이는 역할.

처리 흐름:
  1. 유니버스 종목 코드 + KOSPI 지수 + 섹터 맵 로드
  2. ThreadPoolExecutor로 종목별 _collect_ticker() 병렬 실행
  3. 수집된 후보에 전문가별 휴리스틱 스코어 산정
  4. 스코어 내림차순으로 각 전문가 Top-N 반환
"""

from datetime import datetime, timedelta

from loguru import logger

from config import SETTINGS
from data.collector import KoreanStockCollector
from data.feature_engineer import FeatureEngineer
from data.text_generator import generate_news_text, generate_pattern_text
from vector_store.embedder import embed_single
from vector_store.store import PatternStore

# 전문가당 LLM에 넘길 최대 후보 수 (settings.yaml screener.top_n, 기본 15)
TOP_N = SETTINGS.get("screener", {}).get("top_n", 15)


# ── 전문가별 휴리스틱 스코어 ──────────────────────────────────────────────────
# 각 함수는 snapshot(기술적 지표)과 financials(재무 지표)를 받아 0~8 범위의 점수 반환.
# 높은 점수 = 해당 전문가 관점에서 매력적인 종목.

def _score_growth(snap: dict, fin: dict) -> float:
    """성장주 스코어: 거래량 동반 상승 + MACD 양전환 + RSI 모멘텀 구간 + 실적 YoY"""
    s = 0.0
    # 거래량 비율: 최대 2점 (5배 이상이면 만점)
    s += min(snap.get("volume_ratio", 1) / 5, 1) * 2
    # MACD 양전환 = 골든크로스 상태
    s += 1.0 if snap.get("macd_diff", 0) > 0 else 0.0
    # RSI 45~70: 과매수 직전의 강세 모멘텀 구간
    s += 1.0 if 45 < snap.get("rsi", 50) < 72 else 0.0
    # 5일 수익률 양수 = 단기 상승세 (최대 2점)
    s += min(max(snap.get("ret_5d", 0), 0) * 20, 2)
    # 영업이익 YoY 성장률 (최대 2점)
    oy = fin.get("op_income_yoy")
    ry = fin.get("revenue_yoy")
    if oy is not None:
        s += min(max(float(oy) / 50, 0), 2)   # 50% 성장 = 2점
    if ry is not None:
        s += min(max(float(ry) / 50, 0), 1)   # 매출 50% 성장 = 1점
    return s


def _score_value(snap: dict, fin: dict) -> float:
    """가치주 스코어: 저PBR/PER + 섹터 대비 저평가 + ROE + 낮은 부채비율"""
    s = 0.0
    pbr = float(fin.get("pbr") or 99)
    per = float(fin.get("per") or 99)
    pv_per = float(fin.get("per_vs_sector") or 0)   # 섹터 대비 PER 위치 (음수=저평가)
    pv_pbr = float(fin.get("pbr_vs_sector") or 0)   # 섹터 대비 PBR 위치
    # PBR 0~1.5: 낮을수록 높은 점수 (최대 1.5점)
    if 0 < pbr < 1.5:
        s += 1.5 - pbr
    # PER 0~15: 낮을수록 높은 점수 (최대 3점)
    if 0 < per < 15:
        s += (15 - per) / 5
    # 섹터 대비 저PER (최대 1점)
    if pv_per < 0:
        s += min(abs(pv_per), 0.5) * 2
    # 섹터 대비 저PBR (최대 1점)
    if pv_pbr < 0:
        s += min(abs(pv_pbr), 0.5) * 2
    # ROE 5% 이상: 수익성 있는 저평가 종목 우선
    roe = float(fin.get("roe") or 0)
    if roe > 5:
        s += min(roe / 20, 1)  # 20% ROE = 1점
    # 부채비율 100% 이하 = 재무 안정성
    debt = float(fin.get("debt_ratio") or 200)
    if debt < 100:
        s += 1.0
    return s


def _score_theme(snap: dict, fin: dict) -> float:
    """테마주 스코어: 거래량 폭발 + 단기 급등 + RSI 과열 직전 + 공매도"""
    s = 0.0
    vr = float(snap.get("volume_ratio", 1))
    # 거래량 비율 3배+ = 3점, 2배+ = 2점, 1.5배+ = 1점
    if vr > 3:
        s += 3.0
    elif vr > 2:
        s += 2.0
    elif vr > 1.5:
        s += 1.0
    # 5일 수익률 5%+ = 2점, 2%+ = 1점 (모멘텀 확인)
    ret5 = float(snap.get("ret_5d", 0))
    if ret5 > 0.05:
        s += 2.0
    elif ret5 > 0.02:
        s += 1.0
    # RSI 55~75: 강세 모멘텀이지만 아직 과매수 아님
    rsi = float(snap.get("rsi", 50))
    if 55 < rsi < 75:
        s += 1.0
    # 공매도 비중 높으면 숏스퀴즈 가능성 (추가 모멘텀 촉매)
    sr = float(fin.get("short_ratio") or 0)
    if sr > 5:
        s += 0.5
    return s


def _score_dividend(snap: dict, fin: dict) -> float:
    """배당주 스코어: 고배당수익률 + 적정 배당성향 + FCF + 낮은 베타 + 낮은 부채"""
    s = 0.0
    # 배당수익률 4%+ = 3점, 2%+ = 1점
    div = float(fin.get("div") or 0)
    if div > 4:
        s += 3.0
    elif div > 2:
        s += 1.0
    # 배당성향 0~70%: 지속 가능한 배당 구간
    pr = float(fin.get("payout_ratio") or 100)
    if 0 < pr < 70:
        s += 1.0
    # FCF 양수 = 배당 재원 확보 (현금흐름 기반 배당 가능성)
    fcf = fin.get("fcf")
    if fcf is not None and float(fcf) > 0:
        s += 2.0
    # 베타 < 0.8 = 시장보다 낮은 변동성 (방어적 성격)
    beta = float(snap.get("beta_20d") or 1)
    if 0 < beta < 0.8:
        s += 1.0
    # 부채비율 100% 이하 = 배당 지속 가능성
    debt = float(fin.get("debt_ratio") or 200)
    if debt < 100:
        s += 1.0
    return s


def _score_crisis(snap: dict, fin: dict) -> float:
    """위기종목 스코어: 급락 폭 + RSI 과매도 + BB 하단 + 재무 건전성 (반등 가능성)"""
    s = 0.0
    ret5 = float(snap.get("ret_5d", 0))
    rsi = float(snap.get("rsi", 50))
    bb_pct = float(snap.get("bb_pct", 0.5))
    # 5일 급락: -10%+ = 3점, -5%+ = 1점 (더 많이 떨어질수록 반등 기대)
    if ret5 < -0.10:
        s += 3.0
    elif ret5 < -0.05:
        s += 1.0
    # RSI 과매도: 30 이하 = 3점, 40 이하 = 1점
    if rsi < 30:
        s += 3.0
    elif rsi < 40:
        s += 1.0
    # BB 하단 접근: 통계적 과매도 지지선 (반등 가능성)
    if bb_pct < 0.1:
        s += 2.0
    elif bb_pct < 0.2:
        s += 1.0
    # 재무 건전성: 과매도에서 버틸 수 있는 기초체력
    debt = float(fin.get("debt_ratio") or 0)
    cr = float(fin.get("current_ratio") or 0)
    if 0 < debt < 100:
        s += 1.0  # 낮은 부채비율 = 급락 후 생존 가능성
    if cr > 150:
        s += 1.0  # 높은 유동비율 = 단기 자금 위기 없음
    return s


# 전문가 이름 → 스코어 함수 매핑
_SCORERS = {
    "growth": _score_growth,
    "value": _score_value,
    "theme": _score_theme,
    "dividend": _score_dividend,
    "crisis": _score_crisis,
}


# ── 스크리너 노드 ─────────────────────────────────────────────────────────────

def screener_node(state: dict) -> dict:
    """전체 종목 데이터 수집 → 전문가별 Top-N 후보 반환 (LangGraph 진입 노드)

    병렬 처리로 종목별 OHLCV·재무·공시·RAG 패턴을 수집하고
    전문가별 휴리스틱 스코어 기준 상위 TOP_N개를 각 후보 리스트에 담아 반환한다.
    """
    date = state["date"]
    today_dash = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    # 기술적 지표 계산에 충분한 과거 데이터: 380일치 (252 거래일 + 여유분)
    start = (datetime.strptime(date, "%Y%m%d") - timedelta(days=380)).strftime("%Y%m%d")

    collector = KoreanStockCollector()
    engineer = FeatureEngineer(window=SETTINGS["data"]["feature_window"])
    pattern_store = PatternStore()

    tickers = collector.get_universe()
    kospi_df = collector.get_index_ohlcv("1001", start, date)   # KOSPI 지수 (베타 계산용)
    sector_map = collector.get_sector_map()

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_candidates: list[dict] = []
    lock = threading.Lock()
    counter = {"done": 0}

    logger.info(f"스크리닝 시작: {today_dash} | {len(tickers)}종목 | workers=3")

    # workers=3: API 과부하를 피하면서 병렬 처리 (pykrx 서버 부하 고려)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(
                _collect_ticker,
                ticker, date, today_dash, start,
                kospi_df, sector_map, collector, engineer, pattern_store,
            ): ticker
            for ticker in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                cand = future.result()
                if cand:
                    with lock:
                        all_candidates.append(cand)
            except Exception as e:
                logger.debug(f"{ticker} 수집 실패: {e}")
            with lock:
                counter["done"] += 1
                if counter["done"] % 50 == 0:
                    logger.info(f"  스크리닝 {counter['done']}/{len(tickers)}")

    logger.info(f"데이터 수집 완료: {len(all_candidates)}/{len(tickers)}")

    # state의 rag_label_key를 후보 각각에 전달 (전문가 포맷터가 RAG 섹션 생성 시 사용)
    rag_label_key = state.get("rag_label_key", "label_20d")

    result: dict = {}
    for expert, scorer in _SCORERS.items():
        # 전체 후보에 스코어 산정 후 내림차순 정렬 → Top-N 선택
        scored = [
            (scorer(c["snapshot"], c["financials"]), c)
            for c in all_candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:TOP_N]]
        for c in top:
            c["rag_label_key"] = rag_label_key  # 각 후보에 label 키 주입
        result[f"{expert}_candidates"] = top
        logger.info(f"  {expert}: Top-{len(top)} 후보 선정")

    return result


def _collect_ticker(ticker, date, today_dash, start, kospi_df, sector_map,
                    collector, engineer, pattern_store) -> dict | None:
    """단일 종목 데이터 수집 (스레드풀 내 실행)

    반환 구조:
      ticker, name, sector, market: 종목 메타데이터
      snapshot: 기술적 지표 스냅샷 (스코어링·임베딩·전문가 포맷팅에 사용)
      financials: 재무·공매도·섹터 대비 지표 통합 딕셔너리
      news_text: DART 공시 텍스트 (전문가 프롬프트에 포함)
      retrieved_patterns: ChromaDB 유사 패턴 Top-K (RAG)
      current_text: 현재 종목 패턴 텍스트 (GateNet·전문가 프롬프트에 사용)

    데이터 부족(60 거래일 미만) 또는 스냅샷 생성 실패 시 None 반환.
    """
    df = collector.get_ohlcv(ticker, start, date)
    if df.empty or len(df) < 60:
        return None  # 기술적 지표 계산에 최소 60일 필요

    # 기술적 지표 계산 + KOSPI 대비 상대강도·베타 추가
    df = engineer.compute(df)
    df = engineer.add_relative_strength(df, kospi_df)

    snap = engineer.get_snapshot_vector(df, today_dash)
    if snap is None:
        return None  # 해당 날짜 데이터 없음

    # 종목 메타데이터 (섹터 맵에서 조회, 없으면 ticker 자체를 이름으로 사용)
    name = sector_map.loc[ticker, "name"] if ticker in sector_map.index else ticker
    market = sector_map.loc[ticker, "market"] if ticker in sector_map.index else "KOSPI"
    sector = collector.get_sector_name(ticker, date)

    # 현재 패턴을 텍스트화 → OpenAI 임베딩 → ChromaDB 유사 패턴 검색 (RAG)
    current_text = generate_pattern_text(
        ticker=ticker, name=name, sector=sector, market=market, snapshot=snap
    )
    current_emb = embed_single(current_text)
    retrieved = pattern_store.query(current_emb, top_k=SETTINGS["retrieval"]["top_k"])

    # DART 공시 텍스트 (최근 30일, 없으면 빈 문자열)
    news_text = ""
    filings = collector.get_recent_filings(ticker, days=30, ref_date=date)
    if filings:
        news_text = generate_news_text(ticker, name, filings)

    # 재무 지표 통합: PER/PBR/배당 + 공매도 + 섹터 대비 PER/PBR + 52주 위치 + 시가총액
    financials = collector.get_financials(ticker, date)
    financials.update(collector.get_shorting_data(ticker, date))
    financials.update(collector.get_sector_avg_fundamental(ticker, date))
    financials.update(engineer.get_52w_position(df, today_dash))

    # 시가총액: OHLCV DataFrame의 mktcap 컬럼에서 해당 날짜 이전 최신값 사용
    if "mktcap" in df.columns:
        valid_cap = df[df.index.strftime("%Y-%m-%d") <= today_dash]["mktcap"].dropna()
        if not valid_cap.empty:
            financials["mktcap"] = int(valid_cap.iloc[-1])

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector,
        "market": market,
        "snapshot": snap,
        "financials": financials,
        "news_text": news_text,
        "retrieved_patterns": retrieved,
        "current_text": current_text,
    }
