"""한국 주식 데이터 수집: pykrx + FinanceDataReader + OpenDartReader"""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger
from pykrx import stock

try:
    import OpenDartReader as odr
    DART_AVAILABLE = True
except ImportError:
    DART_AVAILABLE = False

from config import DART_API_KEY, ROOT, SETTINGS

CACHE_DIR = ROOT / SETTINGS["paths"]["data_cache"]
CACHE_DIR.mkdir(exist_ok=True)


class KoreanStockCollector:
    def __init__(self):
        self._dart = None

    @property
    def dart(self):
        if self._dart is None and DART_AVAILABLE and DART_API_KEY:
            self._dart = odr.OpenDartReader(DART_API_KEY)
        return self._dart

    # ── 유니버스 ──────────────────────────────────────────────

    def get_universe(self) -> list[str]:
        """KOSPI200 + KOSDAQ150 종목 코드 반환"""
        tickers = set()
        cfg = SETTINGS["data"]["universe"]

        if cfg.get("kospi200"):
            try:
                tickers.update(stock.get_index_portfolio_deposit_file("1028"))
            except Exception as e:
                logger.warning(f"KOSPI200 로딩 실패: {e}")

        if cfg.get("kosdaq150"):
            try:
                tickers.update(stock.get_index_portfolio_deposit_file("2203"))
            except Exception as e:
                logger.warning(f"KOSDAQ150 로딩 실패: {e}")

        result = sorted(tickers)
        logger.info(f"유니버스: {len(result)}개 종목")
        return result

    # ── OHLCV ────────────────────────────────────────────────

    def get_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """일별 OHLCV + 시가총액 + 외국인 보유율"""
        cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df_price = stock.get_market_ohlcv(start, end, ticker)
        df_price.index = pd.to_datetime(df_price.index)
        df_price.columns = ["open", "high", "low", "close", "volume", "amount", "changes"]

        try:
            df_cap = stock.get_market_cap(start, end, ticker)
            df_cap.index = pd.to_datetime(df_cap.index)
            df_cap = df_cap[["시가총액", "거래대금"]].rename(
                columns={"시가총액": "mktcap", "거래대금": "turnover"}
            )
            df = df_price.join(df_cap, how="left")
        except Exception:
            df = df_price

        df["ticker"] = ticker
        df.to_parquet(cache_path)
        return df

    def get_ohlcv_bulk(self, tickers: list[str], start: str, end: str,
                       delay: float = 0.3) -> dict[str, pd.DataFrame]:
        """여러 종목 일괄 수집 (서버 부하 방지 delay 포함)"""
        result = {}
        for i, ticker in enumerate(tickers):
            try:
                result[ticker] = self.get_ohlcv(ticker, start, end)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{len(tickers)} 수집 완료")
            except Exception as e:
                logger.warning(f"{ticker} 수집 실패: {e}")
            time.sleep(delay)
        return result

    # ── 섹터 ─────────────────────────────────────────────────

    def get_sector_map(self) -> pd.DataFrame:
        """KRX 종목별 섹터(업종) 매핑"""
        cache_path = CACHE_DIR / "sector_map.parquet"
        if cache_path.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if age.days < 7:
                return pd.read_parquet(cache_path)

        today = datetime.today().strftime("%Y%m%d")
        rows = []
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                df = stock.get_market_ticker_name(market=market)
                for ticker, name in df.items():
                    try:
                        sector_df = stock.get_market_sector_classifications(today, market)
                        rows.append({"ticker": ticker, "name": name, "market": market})
                    except Exception:
                        rows.append({"ticker": ticker, "name": name, "market": market})
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"{market} 섹터 로딩 실패: {e}")

        df = pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")
        df.to_parquet(cache_path)
        return df

    def get_sector_name(self, ticker: str, date: str) -> str:
        """특정 날짜 기준 종목 섹터명"""
        try:
            for market in ["KOSPI", "KOSDAQ"]:
                df = stock.get_market_sector_classifications(date, market)
                if ticker in df.index:
                    return df.loc[ticker, "섹터"] if "섹터" in df.columns else "기타"
        except Exception:
            pass
        return "기타"

    # ── DART 공시 ────────────────────────────────────────────

    def get_recent_filings(self, corp_code: str, days: int = 30) -> list[dict]:
        """최근 N일 DART 공시 목록"""
        if self.dart is None:
            return []
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            df = self.dart.list(corp_code, start=start, kind="A", final="Y")
            if df is None or df.empty:
                return []
            return df[["rcept_dt", "report_nm", "corp_name"]].to_dict("records")
        except Exception as e:
            logger.debug(f"DART 공시 조회 실패 ({corp_code}): {e}")
            return []

    def get_financial_summary(self, corp_code: str) -> dict:
        """최근 연간 재무요약 (매출, 영업이익, 순이익)"""
        if self.dart is None:
            return {}
        try:
            df = self.dart.finstate_all(corp_code, 2024, reprt_code="11011")
            if df is None or df.empty:
                return {}
            metrics = {}
            for _, row in df.iterrows():
                acc = row.get("account_nm", "")
                val = row.get("thstrm_amount", "0")
                if "매출" in acc:
                    metrics["revenue"] = val
                elif "영업이익" in acc:
                    metrics["op_income"] = val
                elif "당기순이익" in acc:
                    metrics["net_income"] = val
            return metrics
        except Exception as e:
            logger.debug(f"DART 재무 조회 실패 ({corp_code}): {e}")
            return {}

    # ── 인덱스 ───────────────────────────────────────────────

    def get_index_ohlcv(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """지수 OHLCV (KOSPI: 1001, KOSDAQ: 2001)"""
        df = stock.get_index_ohlcv(start, end, index_code)
        df.index = pd.to_datetime(df.index)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        return df
