"""한국 주식 데이터 수집: pykrx (primary) + FinanceDataReader (fallback)"""

import time
from datetime import datetime, timedelta

import FinanceDataReader as fdr
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

# KRX 지수코드 → FDR 심볼 매핑
_INDEX_FDR = {"1001": "KS11", "2001": "KQ11"}


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
        """KOSPI200 + KOSDAQ150 종목 코드 반환 (pykrx 실패 시 FDR fallback)"""
        tickers = set()
        cfg = SETTINGS["data"]["universe"]

        # 1차: pykrx 인덱스 구성 종목
        if cfg.get("kospi200"):
            tickers.update(self._get_index_tickers("1028", "KOSPI200"))
        if cfg.get("kosdaq150"):
            tickers.update(self._get_index_tickers("2203", "KOSDAQ150"))

        # pykrx 실패 시 FDR로 전체 상장 종목 사용
        if not tickers:
            logger.warning("pykrx 유니버스 실패 → FinanceDataReader fallback")
            tickers.update(self._get_universe_fdr())

        result = sorted(tickers)
        logger.info(f"유니버스: {len(result)}개 종목")
        return result

    def _get_index_tickers(self, index_code: str, name: str) -> list[str]:
        try:
            result = stock.get_index_portfolio_deposit_file(index_code)
            if result:
                return list(result)
            logger.warning(f"{name} pykrx 응답 비어있음")
        except Exception as e:
            logger.warning(f"{name} pykrx 실패: {e}")
        return []

    def _get_universe_fdr(self, max_per_market: int = 200) -> list[str]:
        """FDR로 KOSPI + KOSDAQ 시가총액 상위 종목 반환"""
        tickers = []
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                df = fdr.StockListing(market)
                if df.empty:
                    continue
                sym_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
                tickers.extend(df[sym_col].dropna().astype(str).tolist()[:max_per_market])
                logger.info(f"FDR {market}: {min(len(df), max_per_market)}개 종목 로드")
            except Exception as e:
                logger.warning(f"FDR {market} 실패: {e}")
        return tickers

    # ── OHLCV ────────────────────────────────────────────────

    def get_ohlcv(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """일별 OHLCV — pykrx 실패 시 FDR fallback, 캐시 사용"""
        cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        df = self._get_ohlcv_pykrx(ticker, start, end)
        if df.empty:
            df = self._get_ohlcv_fdr(ticker, start, end)

        if not df.empty:
            df["ticker"] = ticker
            df.to_parquet(cache_path)
        return df

    def _get_ohlcv_pykrx(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            df = stock.get_market_ohlcv(start, end, ticker)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            df.columns = ["open", "high", "low", "close", "volume", "amount", "changes"]
            try:
                df_cap = stock.get_market_cap(start, end, ticker)
                df_cap.index = pd.to_datetime(df_cap.index)
                if "시가총액" in df_cap.columns:
                    df_cap = df_cap[["시가총액", "거래대금"]].rename(
                        columns={"시가총액": "mktcap", "거래대금": "turnover"}
                    )
                    df = df.join(df_cap, how="left")
            except Exception:
                pass
            return df
        except Exception as e:
            logger.debug(f"pykrx OHLCV 실패 ({ticker}): {e}")
            return pd.DataFrame()

    def _get_ohlcv_fdr(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        try:
            start_d = f"{start[:4]}-{start[4:6]}-{start[6:]}"
            end_d = f"{end[:4]}-{end[4:6]}-{end[6:]}"
            df = fdr.DataReader(ticker, start_d, end_d)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            # FDR 컬럼 정규화
            col_map = {
                "Open": "open", "High": "high", "Low": "low",
                "Close": "close", "Volume": "volume",
                "Change": "changes",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            if "amount" not in df.columns:
                df["amount"] = df.get("close", 0) * df.get("volume", 0)
            if "changes" not in df.columns:
                df["changes"] = df["close"].pct_change()
            return df
        except Exception as e:
            logger.debug(f"FDR OHLCV 실패 ({ticker}): {e}")
            return pd.DataFrame()

    def get_ohlcv_bulk(self, tickers: list[str], start: str, end: str,
                       delay: float = 0.3) -> dict[str, pd.DataFrame]:
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
        """KRX 종목별 섹터 매핑 — FDR StockListing 우선 사용"""
        cache_path = CACHE_DIR / "sector_map.parquet"
        if cache_path.exists():
            age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
            if age.days < 7:
                return pd.read_parquet(cache_path)

        rows = []
        for market in ["KOSPI", "KOSDAQ"]:
            try:
                df = fdr.StockListing(market)
                if df.empty:
                    continue
                for _, row in df.iterrows():
                    sym = str(row.get("Symbol", row.get("Code", "")))
                    name = str(row.get("Name", ""))
                    sector = str(row.get("Sector", "기타"))
                    if sym:
                        rows.append({"ticker": sym, "name": name,
                                     "sector": sector, "market": market})
            except Exception as e:
                logger.warning(f"FDR {market} 섹터 로딩 실패: {e}")

        if not rows:
            logger.warning("섹터 맵 로딩 실패, 빈 DataFrame 반환")
            return pd.DataFrame(columns=["name", "sector", "market"])

        df = pd.DataFrame(rows).drop_duplicates("ticker").set_index("ticker")
        df.to_parquet(cache_path)
        logger.info(f"섹터 맵 구축: {len(df)}개 종목")
        return df

    def get_sector_name(self, ticker: str, date: str = None) -> str:
        """종목 섹터명 반환"""
        try:
            sector_map = self.get_sector_map()
            if ticker in sector_map.index:
                return str(sector_map.loc[ticker, "sector"])
        except Exception:
            pass
        return "기타"

    # ── DART 공시 ────────────────────────────────────────────

    def get_recent_filings(self, corp_code: str, days: int = 30) -> list[dict]:
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
        """지수 OHLCV — pykrx 실패 시 FDR fallback"""
        df = self._get_index_pykrx(index_code, start, end)
        if df.empty:
            fdr_symbol = _INDEX_FDR.get(index_code, "KS11")
            df = self._get_ohlcv_fdr(fdr_symbol, start, end)
            logger.info(f"지수 {index_code} → FDR({fdr_symbol}) fallback")
        return df

    def _get_index_pykrx(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        try:
            df = stock.get_index_ohlcv(start, end, index_code)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            return df
        except Exception as e:
            logger.debug(f"pykrx 지수 실패 ({index_code}): {e}")
            return pd.DataFrame()
