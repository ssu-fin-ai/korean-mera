"""메인 파이프라인: 데이터 수집 → 임베딩 → 신호 생성 → 리포트"""

import json
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
from db.supabase_store import PortfolioStore
from portfolio.aggregator import portfolio_to_df

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


class MERAPipeline:
    def __init__(self):
        self.collector = KoreanStockCollector()
        self.engineer = FeatureEngineer(window=SETTINGS["data"]["feature_window"])
        self.pattern_store = PatternStore()
        self.news_store = NewsStore()
        self.portfolio_store = PortfolioStore()

    # ── Phase 1: 히스토리 DB 구축 (최초 1회) ────────────────────────────────

    def build_history_db(self, years: int = None):
        """과거 패턴을 ChromaDB에 적재 (최초 실행 시 1회)"""
        years = years or SETTINGS["data"]["history_years"]
        end = datetime.today().strftime("%Y%m%d")
        start = (datetime.today() - timedelta(days=365 * years)).strftime("%Y%m%d")

        tickers = self.collector.get_universe()
        logger.info(f"히스토리 DB 구축 시작: {len(tickers)}종목, {years}년치")

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

    # ── Phase 2: 일별 신호 생성 (LangGraph) ─────────────────────────────────

    def run_daily(self, date: str = None) -> str:
        """LangGraph MERA 그래프 실행 → 포트폴리오 리포트 반환"""
        today = date or datetime.today().strftime("%Y%m%d")
        today_dash = f"{today[:4]}-{today[4:6]}-{today[6:]}"

        # 이전 포트폴리오 수익률 평가
        self._evaluate_prev_portfolio(today_dash)

        # LangGraph 그래프 실행
        from agents.graph import build_mera_graph
        graph = build_mera_graph()

        logger.info(f"LangGraph MERA 실행: {today_dash}")
        final_state = graph.invoke({"date": today})

        report: str = final_state.get("report", f"[{today_dash}] 리포트 생성 실패")
        portfolio: list[dict] = final_state.get("final_portfolio", [])

        # 포트폴리오 DB 저장 (다음날 평가에 사용)
        if portfolio:
            df = portfolio_to_df(portfolio, today_dash)
            self.portfolio_store.save(today_dash, df)

        # 리포트 텍스트 저장
        report_path = REPORTS_DIR / f"report_{today}.txt"
        report_path.write_text(report, encoding="utf-8")

        # 포트폴리오 JSON 저장
        if portfolio:
            json_path = REPORTS_DIR / f"portfolio_{today}.json"
            json_path.write_text(
                json.dumps(portfolio, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        logger.info(f"리포트 저장: {report_path}")
        return report

    # ── Phase 2.5: 이전 포트폴리오 평가 ─────────────────────────────────────

    def _get_weekly_portfolio_date(self, today_dash: str) -> str | None:
        """5영업일(1주) 전 포트폴리오 날짜 반환"""
        try:
            today_dt = pd.to_datetime(today_dash)
            target_dt = today_dt - pd.tseries.offsets.BDay(5)

            dates = self.portfolio_store.list_dates()
            candidates = [d for d in dates if d < today_dash]
            if not candidates:
                return None

            def _dist(d):
                return abs((pd.to_datetime(d) - target_dt).days)

            closest = min(candidates, key=_dist)
            if _dist(closest) > 10:
                return None
            return closest
        except Exception as e:
            logger.debug(f"주간 포트폴리오 날짜 조회 실패: {e}")
            return None

    def _evaluate_prev_portfolio(self, today_dash: str) -> None:
        """지난 주 포트폴리오 수익률 평가 후 EvaluationStore에 저장 (5영업일 기준)"""
        prev_date = self._get_weekly_portfolio_date(today_dash)
        if not prev_date:
            return
        try:
            from portfolio.evaluator import run_evaluation
            result = run_evaluation(prev_date, today_dash, self.collector)
            if result:
                logger.info(
                    f"주간 포트폴리오 평가 완료 ({prev_date} → {today_dash}): "
                    f"평균수익 {result['avg_return']:+.1%} | "
                    f"적중률 {result['hit_rate']:.0%}"
                )
        except Exception as e:
            logger.warning(f"포트폴리오 평가 실패 ({prev_date}): {e}")

    # ── Phase 3: 데이터 업데이트 (오늘치 추가) ───────────────────────────────

    def update_today(self, today: str = None):
        """오늘 날짜 패턴을 ChromaDB에 추가"""
        today = today or datetime.today().strftime("%Y%m%d")
        today_dash = f"{today[:4]}-{today[4:6]}-{today[6:]}"
        start = (datetime.strptime(today, "%Y%m%d") - timedelta(days=380)).strftime("%Y%m%d")

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
