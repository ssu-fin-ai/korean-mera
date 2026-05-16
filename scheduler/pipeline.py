"""메인 파이프라인: 데이터 수집 → 임베딩 → 신호 생성 → 리포트"""

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS
from data.collector import KoreanStockCollector
from data.feature_engineer import FeatureEngineer
from data.text_generator import generate_news_text, generate_pattern_text
from vector_store.embedder import embed_single, embed_texts
from vector_store.store import NewsStore, PatternStore
from agents.gate_agent import route
from agents.experts import run_experts
from portfolio.aggregator import StockSignal, aggregate, build_portfolio, build_report

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


class MERAPipeline:
    def __init__(self):
        self.collector = KoreanStockCollector()
        self.engineer = FeatureEngineer(window=SETTINGS["data"]["feature_window"])
        self.pattern_store = PatternStore()
        self.news_store = NewsStore()

    # ── Phase 1: 히스토리 DB 구축 (최초 1회) ────────────────────────────────

    def build_history_db(self, years: int = None):
        """과거 패턴을 ChromaDB에 적재 (최초 실행 시 1회)"""
        years = years or SETTINGS["data"]["history_years"]
        end = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y%m%d")

        tickers = self.collector.get_universe()
        logger.info(f"히스토리 DB 구축 시작: {len(tickers)}종목, {years}년치")

        # 지수 데이터 (상대강도 계산용)
        kospi_df = self.collector.get_index_ohlcv("1001", start, end)

        sector_map = self.collector.get_sector_map()

        for i, ticker in enumerate(tickers):
            try:
                df = self.collector.get_ohlcv(ticker, start, end)
                if df.empty or len(df) < 60:
                    continue

                df = self.engineer.compute(df)
                df = self.engineer.add_relative_strength(df, kospi_df)

                name = sector_map.loc[ticker, "name"] if ticker in sector_map.index else ticker
                market = sector_map.loc[ticker, "market"] if ticker in sector_map.index else "KOSPI"

                # 매월 말 날짜만 샘플링 (DB 크기 절약)
                sample_dates = df.resample("ME").last().index
                sample_dates = [d.strftime("%Y-%m-%d") for d in sample_dates if not pd.isna(d)]

                ids, texts, embeddings, metas = [], [], [], []

                for date_str in sample_dates:
                    snap = self.engineer.get_snapshot_vector(df, date_str)
                    if snap is None:
                        continue

                    label_row = df[df.index.strftime("%Y-%m-%d") == date_str]
                    label_5d = float(label_row["label_5d"].iloc[0]) if not label_row.empty else None

                    sector = self.collector.get_sector_name(ticker, date_str.replace("-", ""))

                    text = generate_pattern_text(
                        ticker=ticker, name=name, sector=sector, market=market,
                        snapshot=snap, label_5d=label_5d,
                    )
                    emb = embed_single(text)
                    doc_id = f"{ticker}_{date_str}"

                    ids.append(doc_id)
                    texts.append(text)
                    embeddings.append(emb)
                    metas.append({
                        "ticker": ticker, "name": name, "sector": sector,
                        "market": market, "date": date_str,
                        "label_5d": str(label_5d) if label_5d is not None else "",
                    })

                if ids:
                    self.pattern_store.upsert_batch(ids, texts, embeddings, metas)

                if (i + 1) % 20 == 0:
                    logger.info(f"  DB 구축 {i+1}/{len(tickers)} | 총 {self.pattern_store.count()}건")

            except Exception as e:
                logger.error(f"{ticker} DB 구축 실패: {e}")

        logger.info(f"히스토리 DB 구축 완료: {self.pattern_store.count()}건")

    # ── Phase 2: 일별 신호 생성 ──────────────────────────────────────────────

    def run_daily(self, date: str = None) -> str:
        """오늘 날짜 기준 전 종목 분석 → 포트폴리오 리포트 반환"""
        today = date or datetime.today().strftime("%Y%m%d")
        today_dash = f"{today[:4]}-{today[4:6]}-{today[6:]}"
        start = (datetime.strptime(today, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        tickers = self.collector.get_universe()
        kospi_df = self.collector.get_index_ohlcv("1001", start, today)
        sector_map = self.collector.get_sector_map()

        cfg = SETTINGS["portfolio"]
        pipe_cfg = SETTINGS.get("pipeline", {})
        workers = pipe_cfg.get("workers", 10)
        api_sem = threading.Semaphore(pipe_cfg.get("api_concurrency", 5))
        all_signals: list[StockSignal] = []
        counter = {"done": 0}
        lock = threading.Lock()

        logger.info(f"일별 분석 시작: {today_dash} | {len(tickers)}종목 | workers={workers}")

        def analyze_with_sem(ticker):
            with api_sem:
                return self._analyze_one(ticker, today, today_dash, start, kospi_df, sector_map)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_with_sem, t): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    signal = future.result()
                    if signal:
                        with lock:
                            all_signals.append(signal)
                except Exception as e:
                    logger.warning(f"{ticker} 분석 실패: {e}")
                with lock:
                    counter["done"] += 1
                    if counter["done"] % 50 == 0:
                        logger.info(f"  {counter['done']}/{len(tickers)} 분석 완료")

        portfolio = build_portfolio(
            all_signals,
            top_n=cfg["top_n"],
            min_confidence=cfg["signal_threshold"],
        )
        report = build_report(portfolio, today_dash)

        # 리포트 저장
        report_path = REPORTS_DIR / f"report_{today}.txt"
        report_path.write_text(report, encoding="utf-8")
        portfolio.to_csv(REPORTS_DIR / f"portfolio_{today}.csv",
                         index=True, encoding="utf-8-sig")

        logger.info(f"리포트 저장: {report_path}")
        return report

    def _analyze_one(self, ticker, today, today_dash, start, kospi_df, sector_map) -> StockSignal | None:
        df = self.collector.get_ohlcv(ticker, start, today)
        if df.empty or len(df) < 60:
            return None

        df = self.engineer.compute(df)
        df = self.engineer.add_relative_strength(df, kospi_df)

        snap = self.engineer.get_snapshot_vector(df, today_dash)
        if snap is None:
            return None

        name = sector_map.loc[ticker, "name"] if ticker in sector_map.index else ticker
        market = sector_map.loc[ticker, "market"] if ticker in sector_map.index else "KOSPI"
        sector = self.collector.get_sector_name(ticker, today)

        current_text = generate_pattern_text(
            ticker=ticker, name=name, sector=sector, market=market, snapshot=snap
        )

        # 벡터 DB에서 유사 패턴 검색
        current_emb = embed_single(current_text)
        retrieved = self.pattern_store.query(
            current_emb,
            top_k=SETTINGS["retrieval"]["top_k"],
        )

        # GateNet 라우팅
        gate_result = route(current_text, retrieved)
        expert_names = gate_result.get("experts", ["growth"])

        # 뉴스 컨텍스트 (DART)
        news_text = ""
        filings = self.collector.get_recent_filings(ticker, days=30)
        if filings:
            news_text = generate_news_text(ticker, name, filings)

        # 전문가 에이전트 실행
        expert_results = run_experts(
            expert_names, current_text, retrieved, news_text
        )

        stock_signal = StockSignal(
            ticker=ticker, name=name, sector=sector, date=today_dash,
            gate_result=gate_result, expert_results=expert_results,
        )
        return aggregate(stock_signal)

    # ── Phase 3: 데이터 업데이트 (오늘치 추가) ───────────────────────────────

    def update_today(self, today: str = None):
        """오늘 날짜 패턴을 DB에 추가"""
        today = today or datetime.today().strftime("%Y%m%d")
        today_dash = f"{today[:4]}-{today[4:6]}-{today[6:]}"
        start = (datetime.strptime(today, "%Y%m%d") - timedelta(days=200)).strftime("%Y%m%d")

        tickers = self.collector.get_universe()
        kospi_df = self.collector.get_index_ohlcv("1001", start, today)
        sector_map = self.collector.get_sector_map()

        added = 0
        for ticker in tickers:
            try:
                df = self.collector.get_ohlcv(ticker, start, today)
                if df.empty or len(df) < 60:
                    continue
                df = self.engineer.compute(df)
                df = self.engineer.add_relative_strength(df, kospi_df)
                snap = self.engineer.get_snapshot_vector(df, today_dash)
                if snap is None:
                    continue

                name = sector_map.loc[ticker, "name"] if ticker in sector_map.index else ticker
                market = sector_map.loc[ticker, "market"] if ticker in sector_map.index else "KOSPI"
                sector = self.collector.get_sector_name(ticker, today)

                label_row = df[df.index.strftime("%Y-%m-%d") == today_dash]
                label_5d = float(label_row["label_5d"].iloc[0]) if not label_row.empty else None

                text = generate_pattern_text(
                    ticker=ticker, name=name, sector=sector, market=market,
                    snapshot=snap, label_5d=label_5d,
                )
                emb = embed_single(text)
                self.pattern_store.upsert(
                    doc_id=f"{ticker}_{today_dash}",
                    text=text, embedding=emb,
                    metadata={
                        "ticker": ticker, "name": name, "sector": sector,
                        "market": market, "date": today_dash,
                        "label_5d": str(label_5d) if label_5d else "",
                    },
                )
                added += 1
            except Exception as e:
                logger.debug(f"{ticker} 업데이트 실패: {e}")

        logger.info(f"DB 업데이트 완료: {added}건 추가")
