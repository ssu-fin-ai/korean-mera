"""메인 파이프라인: 데이터 수집 → 임베딩 → 신호 생성 → 리포트

3단계 파이프라인:
  Phase 1 (build_history_db): 과거 패턴을 ChromaDB에 적재 (최초 1회)
  Phase 2 (run_daily):        LangGraph MERA 그래프 실행 → 포트폴리오 리포트
  Phase 2.5 (evaluate):       지난주 포트폴리오 실제 수익률 평가
  Phase 3 (update_today):     오늘 패턴을 ChromaDB에 추가 (매일 갱신)
"""

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
from vector_store.store import PatternStore
from db.supabase_store import PortfolioStore
from portfolio.aggregator import portfolio_to_df

# 리포트 저장 디렉토리 (없으면 자동 생성)
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


class MERAPipeline:
    def __init__(self):
        self.collector = KoreanStockCollector()
        self.engineer = FeatureEngineer(window=SETTINGS["data"]["feature_window"])
        self.pattern_store = PatternStore()
        self.portfolio_store = PortfolioStore()

    # ── Phase 1: 히스토리 DB 구축 (최초 1회) ────────────────────────────────

    def build_history_db(self, years: int = None):
        """과거 패턴을 ChromaDB에 적재 (최초 실행 시 1회)

        years년치 OHLCV에서 모든 거래일의 패턴 텍스트를 생성해
        임베딩 벡터와 함께 ChromaDB에 upsert한다.
        이후 run_daily()에서 RAG 검색의 기반 데이터로 활용된다.

        각 문서 ID: "{ticker}_{YYYY-MM-DD}"
        메타데이터: ticker, name, sector, market, date, label_5d/10d/20d
        """
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

                # 모든 거래일에 대해 패턴 텍스트 + 임베딩 생성
                sample_dates = [d.strftime("%Y-%m-%d") for d in df.index if not pd.isna(d)]

                ids, texts, embeddings, metas = [], [], [], []

                for date_str in sample_dates:
                    snap = self.engineer.get_snapshot_vector(df, date_str)
                    if snap is None:
                        continue

                    # 해당 날짜의 미래 수익률 레이블 추출 (RAG 패턴의 정답 데이터)
                    label_row = df[df.index.strftime("%Y-%m-%d") == date_str]
                    def _lbl(col):
                        if label_row.empty or col not in label_row.columns:
                            return None
                        v = label_row[col].iloc[0]
                        return float(v) if not pd.isna(v) else None
                    label_5d  = _lbl("label_5d")
                    label_10d = _lbl("label_10d")
                    label_20d = _lbl("label_20d")

                    sector = self.collector.get_sector_name(ticker, date_str.replace("-", ""))

                    # label_5d 포함 → 이 패턴 이후 실제 수익률이 텍스트에 기록됨
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
                        # ChromaDB는 None을 허용하지 않아 빈 문자열로 저장
                        "label_5d":  str(label_5d)  if label_5d  is not None else "",
                        "label_10d": str(label_10d) if label_10d is not None else "",
                        "label_20d": str(label_20d) if label_20d is not None else "",
                    })

                if ids:
                    self.pattern_store.upsert_batch(ids, texts, embeddings, metas)

                if (i + 1) % 20 == 0:
                    logger.info(f"  DB 구축 {i+1}/{len(tickers)} | 총 {self.pattern_store.count()}건")

            except Exception as e:
                logger.error(f"{ticker} DB 구축 실패: {e}")

        logger.info(f"히스토리 DB 구축 완료: {self.pattern_store.count()}건")

    # ── Phase 2: 일별 신호 생성 (LangGraph) ─────────────────────────────────

    def run_daily(self, date: str = None, horizon: str = "monthly") -> str:
        """LangGraph MERA 그래프 실행 → 포트폴리오 리포트 반환

        horizon 파라미터:
          "weekly"  → rag_label_key="label_5d"  (5일 후 수익률 기준 RAG)
          "monthly" → rag_label_key="label_20d" (20일 후 수익률 기준 RAG, 기본값)

        결과 저장:
          reports/report_{date}.txt    - 텍스트 리포트
          reports/portfolio_{date}.json - 포트폴리오 JSON
          Supabase portfolio_history    - 다음날 수익률 평가를 위한 DB 저장
        """
        today = date or datetime.today().strftime("%Y%m%d")
        today_dash = f"{today[:4]}-{today[4:6]}-{today[6:]}"
        rag_label_key = "label_5d" if horizon == "weekly" else "label_20d"

        # 매일 실행 시 지난주 포트폴리오 수익률 자동 평가
        self._evaluate_prev_portfolio(today_dash)

        # LangGraph 그래프 빌드 및 실행
        from agents.graph import build_mera_graph
        graph = build_mera_graph()

        logger.info(f"LangGraph MERA 실행: {today_dash} | horizon={horizon} ({rag_label_key})")
        final_state = graph.invoke({"date": today, "rag_label_key": rag_label_key})

        report: str = final_state.get("report", f"[{today_dash}] 리포트 생성 실패")
        portfolio: list[dict] = final_state.get("final_portfolio", [])

        # Supabase에 포트폴리오 저장 (5영업일 후 수익률 평가에 사용)
        if portfolio:
            df = portfolio_to_df(portfolio, today_dash)
            self.portfolio_store.save(today_dash, df)

        # 리포트 텍스트 파일 저장
        report_path = REPORTS_DIR / f"report_{today}.txt"
        report_path.write_text(report, encoding="utf-8")

        # 포트폴리오 JSON 저장 (대시보드·백테스트 분석용)
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
        """5영업일(1주) 전 포트폴리오 날짜 반환

        DB에 저장된 날짜 중 '5영업일 전' 날짜와 가장 가까운 것을 찾는다.
        10일 이상 차이나면 신뢰할 수 없으므로 None 반환.
        """
        try:
            today_dt = pd.to_datetime(today_dash)
            target_dt = today_dt - pd.tseries.offsets.BDay(5)  # 5영업일 전

            dates = self.portfolio_store.list_dates()
            candidates = [d for d in dates if d < today_dash]
            if not candidates:
                return None

            def _dist(d):
                return abs((pd.to_datetime(d) - target_dt).days)

            closest = min(candidates, key=_dist)
            if _dist(closest) > 10:  # 10일 이상 차이면 매칭 실패
                return None
            return closest
        except Exception as e:
            logger.debug(f"주간 포트폴리오 날짜 조회 실패: {e}")
            return None

    def _evaluate_prev_portfolio(self, today_dash: str) -> None:
        """지난주 포트폴리오 수익률 평가 후 EvaluationStore에 저장 (5영업일 기준)

        평가 가능한 이전 포트폴리오가 없으면 조용히 스킵.
        평가 결과는 Supabase portfolio_eval_summary / portfolio_eval_stocks에 저장.
        """
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
        """오늘 날짜 패턴을 ChromaDB에 추가

        run_daily() 전에 호출해 최신 데이터를 RAG DB에 반영한다.
        label은 현재 시점에서 미래 값이 없으므로 빈 문자열로 저장.
        나중에 해당 날짜가 지나면 label 값이 채워진 패턴으로 갱신된다.
        """
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

                # 현재 시점의 label: 미래 수익률 (오늘 이후 데이터가 있으면 채워짐)
                label_row = df[df.index.strftime("%Y-%m-%d") == today_dash]
                def _lbl(col):
                    if label_row.empty or col not in label_row.columns:
                        return None
                    v = label_row[col].iloc[0]
                    return float(v) if not pd.isna(v) else None
                label_5d  = _lbl("label_5d")
                label_10d = _lbl("label_10d")
                label_20d = _lbl("label_20d")

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
                        "label_5d":  str(label_5d)  if label_5d  is not None else "",
                        "label_10d": str(label_10d) if label_10d is not None else "",
                        "label_20d": str(label_20d) if label_20d is not None else "",
                    },
                )
                added += 1
            except Exception as e:
                logger.debug(f"{ticker} 업데이트 실패: {e}")

        logger.info(f"DB 업데이트 완료: {added}건 추가")
