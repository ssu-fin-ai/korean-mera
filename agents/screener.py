"""사전 스크리닝 노드: 전체 종목 데이터 수집 + 전문가별 후보 필터링"""

from datetime import datetime, timedelta

from loguru import logger

from config import SETTINGS
from data.collector import KoreanStockCollector
from data.feature_engineer import FeatureEngineer
from data.text_generator import generate_news_text, generate_pattern_text
from vector_store.embedder import embed_single
from vector_store.store import PatternStore

TOP_N = SETTINGS.get("screener", {}).get("top_n", 15)


# ── 전문가별 휴리스틱 스코어 ──────────────────────────────────────────────────

def _score_growth(snap: dict, fin: dict) -> float:
    s = 0.0
    s += min(snap.get("volume_ratio", 1) / 5, 1) * 2
    s += 1.0 if snap.get("macd_diff", 0) > 0 else 0.0
    s += 1.0 if 45 < snap.get("rsi", 50) < 72 else 0.0
    s += min(max(snap.get("ret_5d", 0), 0) * 20, 2)
    oy = fin.get("op_income_yoy")
    ry = fin.get("revenue_yoy")
    if oy is not None:
        s += min(max(float(oy) / 50, 0), 2)
    if ry is not None:
        s += min(max(float(ry) / 50, 0), 1)
    return s


def _score_value(snap: dict, fin: dict) -> float:
    s = 0.0
    pbr = float(fin.get("pbr") or 99)
    per = float(fin.get("per") or 99)
    pv_per = float(fin.get("per_vs_sector") or 0)
    pv_pbr = float(fin.get("pbr_vs_sector") or 0)
    if 0 < pbr < 1.5:
        s += 1.5 - pbr
    if 0 < per < 15:
        s += (15 - per) / 5
    if pv_per < 0:
        s += min(abs(pv_per), 0.5) * 2
    if pv_pbr < 0:
        s += min(abs(pv_pbr), 0.5) * 2
    roe = float(fin.get("roe") or 0)
    if roe > 5:
        s += min(roe / 20, 1)
    debt = float(fin.get("debt_ratio") or 200)
    if debt < 100:
        s += 1.0
    return s


def _score_theme(snap: dict, fin: dict) -> float:
    s = 0.0
    vr = float(snap.get("volume_ratio", 1))
    if vr > 3:
        s += 3.0
    elif vr > 2:
        s += 2.0
    elif vr > 1.5:
        s += 1.0
    ret5 = float(snap.get("ret_5d", 0))
    if ret5 > 0.05:
        s += 2.0
    elif ret5 > 0.02:
        s += 1.0
    rsi = float(snap.get("rsi", 50))
    if 55 < rsi < 75:
        s += 1.0
    sr = float(fin.get("short_ratio") or 0)
    if sr > 5:
        s += 0.5
    return s


def _score_dividend(snap: dict, fin: dict) -> float:
    s = 0.0
    div = float(fin.get("div") or 0)
    if div > 4:
        s += 3.0
    elif div > 2:
        s += 1.0
    pr = float(fin.get("payout_ratio") or 100)
    if 0 < pr < 70:
        s += 1.0
    fcf = fin.get("fcf")
    if fcf is not None and float(fcf) > 0:
        s += 2.0
    beta = float(snap.get("beta_20d") or 1)
    if 0 < beta < 0.8:
        s += 1.0
    debt = float(fin.get("debt_ratio") or 200)
    if debt < 100:
        s += 1.0
    return s


def _score_crisis(snap: dict, fin: dict) -> float:
    """과매도 + 재무 안전 = 반등 가능성"""
    s = 0.0
    ret5 = float(snap.get("ret_5d", 0))
    rsi = float(snap.get("rsi", 50))
    bb_pct = float(snap.get("bb_pct", 0.5))
    if ret5 < -0.10:
        s += 3.0
    elif ret5 < -0.05:
        s += 1.0
    if rsi < 30:
        s += 3.0
    elif rsi < 40:
        s += 1.0
    if bb_pct < 0.1:
        s += 2.0
    elif bb_pct < 0.2:
        s += 1.0
    debt = float(fin.get("debt_ratio") or 0)
    cr = float(fin.get("current_ratio") or 0)
    if 0 < debt < 100:
        s += 1.0
    if cr > 150:
        s += 1.0
    return s


_SCORERS = {
    "growth": _score_growth,
    "value": _score_value,
    "theme": _score_theme,
    "dividend": _score_dividend,
    "crisis": _score_crisis,
}


# ── 스크리너 노드 ─────────────────────────────────────────────────────────────

def screener_node(state: dict) -> dict:
    """전체 종목 데이터 수집 → 전문가별 Top-N 후보 반환"""
    date = state["date"]
    today_dash = f"{date[:4]}-{date[4:6]}-{date[6:]}"
    start = (datetime.strptime(date, "%Y%m%d") - timedelta(days=380)).strftime("%Y%m%d")

    collector = KoreanStockCollector()
    engineer = FeatureEngineer(window=SETTINGS["data"]["feature_window"])
    pattern_store = PatternStore()

    tickers = collector.get_universe()
    kospi_df = collector.get_index_ohlcv("1001", start, date)
    sector_map = collector.get_sector_map()

    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    all_candidates: list[dict] = []
    lock = threading.Lock()
    counter = {"done": 0}

    logger.info(f"스크리닝 시작: {today_dash} | {len(tickers)}종목 | workers=3")

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

    rag_label_key = state.get("rag_label_key", "label_20d")

    result: dict = {}
    for expert, scorer in _SCORERS.items():
        scored = [
            (scorer(c["snapshot"], c["financials"]), c)
            for c in all_candidates
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scored[:TOP_N]]
        for c in top:
            c["rag_label_key"] = rag_label_key
        result[f"{expert}_candidates"] = top
        logger.info(f"  {expert}: Top-{len(top)} 후보 선정")

    return result


def _collect_ticker(ticker, date, today_dash, start, kospi_df, sector_map,
                    collector, engineer, pattern_store) -> dict | None:
    df = collector.get_ohlcv(ticker, start, date)
    if df.empty or len(df) < 60:
        return None

    df = engineer.compute(df)
    df = engineer.add_relative_strength(df, kospi_df)

    snap = engineer.get_snapshot_vector(df, today_dash)
    if snap is None:
        return None

    name = sector_map.loc[ticker, "name"] if ticker in sector_map.index else ticker
    market = sector_map.loc[ticker, "market"] if ticker in sector_map.index else "KOSPI"
    sector = collector.get_sector_name(ticker, date)

    current_text = generate_pattern_text(
        ticker=ticker, name=name, sector=sector, market=market, snapshot=snap
    )
    current_emb = embed_single(current_text)
    retrieved = pattern_store.query(current_emb, top_k=SETTINGS["retrieval"]["top_k"])

    news_text = ""
    filings = collector.get_recent_filings(ticker, days=30, ref_date=date)
    if filings:
        news_text = generate_news_text(ticker, name, filings)

    financials = collector.get_financials(ticker, date)
    financials.update(collector.get_shorting_data(ticker, date))
    financials.update(collector.get_sector_avg_fundamental(ticker, date))
    financials.update(engineer.get_52w_position(df, today_dash))

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
