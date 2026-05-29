"""한국 주식 데이터 수집: pykrx (primary) + FinanceDataReader (fallback)"""

import json
import statistics
import time
from datetime import datetime, timedelta

import FinanceDataReader as fdr
import pandas as pd
from loguru import logger
from pykrx import stock

try:
    import opendartreader as odr
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
        self._dart_code_map: dict[str, str] = {}  # ticker → DART 8자리 법인코드

    @property
    def dart(self):
        if self._dart is None and DART_AVAILABLE and DART_API_KEY:
            self._dart = odr.OpenDartReader(DART_API_KEY)
        return self._dart

    @staticmethod
    def _dart_year_for_date(date: str) -> tuple[int, str]:
        """날짜 기준 가장 최근 제출된 DART 보고서 (year, reprt_code).

        사업보고서 제출 기한: 3월 31일.
        4월 이후 → 전년도 사업보고서, 1~3월 → 전전년도 사업보고서.
        """
        dt = datetime.strptime(date, "%Y%m%d")
        if dt.month >= 4:
            return dt.year - 1, "11011"
        return dt.year - 2, "11011"

    def _ticker_to_dart_code(self, ticker: str) -> str | None:
        """주식 티커(6자리) → DART 법인코드(8자리) 변환 — 인스턴스 캐시"""
        if self.dart is None:
            return None
        if ticker in self._dart_code_map:
            return self._dart_code_map[ticker]
        try:
            codes_df = self.dart.corp_codes
            if codes_df is None or codes_df.empty:
                return None
            mask = codes_df["stock_code"].astype(str).str.zfill(6) == ticker.zfill(6)
            rows = codes_df[mask]
            if not rows.empty:
                code = str(rows.iloc[0]["corp_code"])
                self._dart_code_map[ticker] = code
                return code
        except Exception as e:
            logger.debug(f"DART 법인코드 조회 실패 ({ticker}): {e}")
        return None

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
        """FDR로 KOSPI 시가총액 상위 종목 반환 (KOSDAQ 제외)"""
        tickers = []
        for market in ["KOSPI"]:
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
            ncols = len(df.columns)
            if ncols >= 7:
                df.columns = ["open", "high", "low", "close", "volume", "amount", "changes"] + list(df.columns[7:])
            elif ncols == 6:
                df.columns = ["open", "high", "low", "close", "volume", "changes"]
                df["amount"] = df["close"] * df["volume"]
            elif ncols >= 5:
                df.columns = ["open", "high", "low", "close", "volume"] + list(df.columns[5:])
                df["amount"] = df["close"] * df["volume"]
                df["changes"] = df["close"].pct_change()
            else:
                return pd.DataFrame()
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

    def get_recent_filings(self, ticker: str, days: int = 30) -> list[dict]:
        if self.dart is None:
            return []
        corp_code = self._ticker_to_dart_code(ticker) or ticker
        start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            df = self.dart.list(corp_code, start=start, kind="A", final="Y")
            if df is None or df.empty:
                return []
            return df[["rcept_dt", "report_nm", "corp_name"]].to_dict("records")
        except Exception as e:
            logger.debug(f"DART 공시 조회 실패 ({corp_code}): {e}")
            return []

    def get_financials(self, ticker: str, date: str) -> dict:
        """PER/PBR/배당수익률(pykrx) + 매출/이익(DART) 통합 재무 데이터.

        월 단위로 캐시 — 같은 달 재호출 시 API 생략.
        """
        import json
        cache_path = CACHE_DIR / f"fin_{ticker}_{date[:6]}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        result: dict = {}

        # pykrx: 역사적 날짜별 PER, PBR, EPS, BPS, DIV, DPS (KRX 로그인 필요)
        try:
            start_7d = (datetime.strptime(date, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
            df = stock.get_market_fundamental(start_7d, date, ticker)
            if not df.empty:
                row = df.iloc[-1]
                result.update({
                    "per": round(float(row.get("PER", 0) or 0), 2),
                    "pbr": round(float(row.get("PBR", 0) or 0), 2),
                    "eps": int(row.get("EPS", 0) or 0),
                    "bps": int(row.get("BPS", 0) or 0),
                    "div": round(float(row.get("DIV", 0) or 0), 2),
                    "dps": int(row.get("DPS", 0) or 0),
                })
        except Exception as e:
            logger.debug(f"pykrx fundamental 실패 ({ticker}): {e}")

        # pykrx 실패 시 네이버 금융으로 fallback (현재값만, 백데이터 미지원)
        if not result:
            try:
                result.update(self._get_naver_fundamental(ticker))
            except Exception as e:
                logger.debug(f"네이버 fundamental fallback 실패 ({ticker}): {e}")

        # DART: 매출/이익 YoY + ROA/ROE + 유동비율 + 이자보상배율
        dart_data = self.get_financial_summary(ticker, date)
        result.update(dart_data)

        # ROE (DART 우선, 없으면 EPS/BPS 근사)
        eps = result.get("eps", 0)
        bps = result.get("bps", 0)
        dps = result.get("dps", 0)
        if "roe_dart" in result:
            result["roe"] = result.pop("roe_dart")
        elif eps and bps and bps > 0:
            result["roe"] = round(float(eps) / float(bps) * 100, 2)

        # 배당성향 (payout ratio)
        if eps and dps and eps > 0:
            result["payout_ratio"] = round(float(dps) / float(eps) * 100, 1)

        # Graham Number = √(22.5 × EPS × BPS)
        import math
        if eps and bps and eps > 0 and bps > 0:
            result["graham_number"] = round(math.sqrt(22.5 * float(eps) * float(bps)), 0)

        if result:
            try:
                cache_path.write_text(
                    json.dumps(result, ensure_ascii=False), encoding="utf-8"
                )
            except Exception:
                pass

        return result

    def _get_naver_fundamental(self, ticker: str) -> dict:
        """네이버 금융 API에서 PER/PBR/EPS/BPS/배당수익률 조회 (주말 포함 항상 동작)"""
        import urllib.request as ur
        import re

        url = f"https://m.stock.naver.com/api/stock/{ticker}/integration"
        req = ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with ur.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        def _parse(val: str) -> float:
            cleaned = re.sub(r"[^\d.]", "", str(val).replace(",", ""))
            return float(cleaned) if cleaned else 0.0

        code_map = {
            "per": "per", "pbr": "pbr", "eps": "eps",
            "bps": "bps", "dividendYieldRatio": "div",
        }
        result: dict = {}
        for item in data.get("totalInfos", []):
            code = item.get("code", "")
            if code in code_map:
                result[code_map[code]] = _parse(item.get("value", "0"))
        return result

    def get_financial_summary(self, ticker: str, date: str = "20250101") -> dict:
        if self.dart is None:
            return {}
        corp_code = self._ticker_to_dart_code(ticker)
        if not corp_code:
            return {}
        year, reprt_code = self._dart_year_for_date(date)
        try:
            df = self.dart.finstate_all(corp_code, year, reprt_code=reprt_code)
            if df is None or df.empty:
                return {}

            def _to_int(val) -> int:
                try:
                    return int(str(val).replace(",", "").strip())
                except Exception:
                    return 0

            def _yoy(curr: int, prev: int) -> float | None:
                if prev and prev != 0:
                    return round((curr - prev) / abs(prev) * 100, 1)
                return None

            metrics: dict = {}
            _ni = _ni_prev = _ta = _te = _td = _ocf = _capex = 0
            _rev_prev = _opi_prev = _opi_num = 0
            _ca = _cl = _int_exp = 0  # 유동자산, 유동부채, 이자비용

            for _, row in df.iterrows():
                acc = str(row.get("account_nm", ""))
                sj  = str(row.get("sj_div", ""))
                raw  = str(row.get("thstrm_amount", "0"))
                num  = _to_int(raw)
                prev = _to_int(row.get("frmtrm_amount", "0"))

                # ── 손익계산서 ────────────────────────────────────
                if sj in ("IS", "CIS"):
                    if "매출" in acc and "원가" not in acc and "비용" not in acc:
                        metrics["revenue"] = raw
                        _rev_prev = prev
                    elif "영업이익" in acc and "손실" not in acc:
                        metrics["op_income"] = raw
                        _opi_prev = prev
                        _opi_num = num
                    elif "당기순이익" in acc and "비지배" not in acc and "net_income" not in metrics:
                        metrics["net_income"] = raw
                        _ni, _ni_prev = num, prev
                    elif "이자비용" in acc and _int_exp == 0:
                        _int_exp = abs(num)

                # ── 재무상태표 ────────────────────────────────────
                elif sj == "BS":
                    if acc in ("자산총계", "총자산"):
                        _ta = num
                    elif acc in ("부채총계", "총부채"):
                        _td = num
                    elif acc in ("자본총계", "자기자본합계", "자기자본"):
                        _te = num
                    elif acc == "유동자산":
                        _ca = num
                    elif acc == "유동부채":
                        _cl = num

                # ── 현금흐름표 ────────────────────────────────────
                elif sj == "CF":
                    if "영업활동" in acc and "현금" in acc and _ocf == 0:
                        _ocf = num
                    elif "유형자산" in acc and "취득" in acc:
                        _capex = num  # 음수(유출)

            # YoY 성장률
            if "revenue" in metrics:
                yoy = _yoy(_to_int(metrics["revenue"]), _rev_prev)
                if yoy is not None:
                    metrics["revenue_yoy"] = yoy
            if "op_income" in metrics:
                yoy = _yoy(_to_int(metrics["op_income"]), _opi_prev)
                if yoy is not None:
                    metrics["op_income_yoy"] = yoy
            if "net_income" in metrics:
                yoy = _yoy(_ni, _ni_prev)
                if yoy is not None:
                    metrics["net_income_yoy"] = yoy

            # ROA / ROE
            if _ni and _ta > 0:
                metrics["roa"] = round(_ni / _ta * 100, 2)
            if _ni and _te > 0:
                metrics["roe_dart"] = round(_ni / _te * 100, 2)

            # 부채비율
            if _td and _te > 0:
                metrics["debt_ratio"] = round(_td / _te * 100, 1)

            # 유동비율
            if _ca and _cl > 0:
                metrics["current_ratio"] = round(_ca / _cl * 100, 1)

            # 이자보상배율
            if _opi_num and _int_exp > 0:
                metrics["interest_coverage"] = round(_opi_num / _int_exp, 1)

            # FCF = 영업활동현금흐름 - CAPEX (DART는 CAPEX를 양수로 보고)
            if _ocf:
                metrics["ocf"] = _ocf
                if _capex:
                    metrics["fcf"] = _ocf - abs(_capex)

            return metrics
        except Exception as e:
            logger.debug(f"DART 재무 조회 실패 ({corp_code}): {e}")
            return {}

    # ── 투자자별 순매수 ──────────────────────────────────────────

    def get_investor_trading(self, ticker: str, date: str) -> dict:
        """5일 기관/외국인 순매수 수량 + 외국인 보유비율 (네이버 금융) — 일별 캐시"""
        import urllib.request as ur

        cache_path = CACHE_DIR / f"inv_{ticker}_{date}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        result: dict = {}
        try:
            url = f"https://m.stock.naver.com/api/stock/{ticker}/integration"
            req = ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with ur.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())

            def _parse_q(s: str) -> int:
                try:
                    return int(str(s).replace(",", "").strip())
                except Exception:
                    return 0

            deal = data.get("dealTrendInfos", [])
            if deal:
                result["inst_net_5d"] = sum(_parse_q(d.get("organPureBuyQuant", "0")) for d in deal)
                result["foreign_net_5d"] = sum(_parse_q(d.get("foreignerPureBuyQuant", "0")) for d in deal)
                try:
                    hold = deal[0].get("foreignerHoldRatio", "0%").replace("%", "")
                    result["foreign_hold_ratio"] = round(float(hold), 2)
                except Exception:
                    pass
        except Exception as e:
            logger.debug(f"투자자별 거래 조회 실패 ({ticker}): {e}")

        if result:
            try:
                cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return result

    # ── 공매도 ───────────────────────────────────────────────────

    def get_shorting_data(self, ticker: str, date: str) -> dict:
        """공매도 비중·5일 평균 (pykrx) — 일별 캐시"""
        cache_path = CACHE_DIR / f"short_{ticker}_{date}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        result: dict = {}
        try:
            start = (datetime.strptime(date, "%Y%m%d") - timedelta(days=14)).strftime("%Y%m%d")
            df = stock.get_shorting_volume_by_date(start, date, ticker)
            if not df.empty:
                ratio_col = next((c for c in df.columns if "비중" in c), None)
                if ratio_col:
                    result["short_ratio"] = round(float(df[ratio_col].iloc[-1]), 2)
                    result["short_ratio_5d_avg"] = round(float(df[ratio_col].tail(5).mean()), 2)
        except Exception as e:
            logger.debug(f"공매도 데이터 조회 실패 ({ticker}): {e}")

        if result:
            try:
                cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
            except Exception:
                pass
        return result

    # ── 섹터 평균 PER/PBR ─────────────────────────────────────

    def get_sector_avg_fundamental(self, ticker: str, date: str) -> dict:
        """섹터 동종 종목 PER/PBR 중앙값 + 현 종목 상대 위치 (캐시 활용)"""
        sector_map = self.get_sector_map()
        if ticker not in sector_map.index:
            return {}

        sector = str(sector_map.loc[ticker, "sector"])
        if not sector or sector in ("기타", "nan"):
            return {}

        cache_key = sector.replace(" ", "_").replace("/", "_")
        cache_path = CACHE_DIR / f"sector_avg_{cache_key}_{date[:6]}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            peers = sector_map[sector_map["sector"] == sector].index.tolist()
            pers, pbrs = [], []
            for peer in peers:
                fin_file = CACHE_DIR / f"fin_{peer}_{date[:6]}.json"
                if not fin_file.exists():
                    continue
                try:
                    fin = json.loads(fin_file.read_text(encoding="utf-8"))
                    per = float(fin.get("per", 0) or 0)
                    pbr = float(fin.get("pbr", 0) or 0)
                    if 0 < per < 200:
                        pers.append(per)
                    if 0 < pbr < 50:
                        pbrs.append(pbr)
                except Exception:
                    pass

            cached = {}
            if len(pers) >= 3:
                cached["sector_per_median"] = round(statistics.median(pers), 2)
                cached["sector_per_count"] = len(pers)
            if len(pbrs) >= 3:
                cached["sector_pbr_median"] = round(statistics.median(pbrs), 2)

            if cached:
                try:
                    cache_path.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
                except Exception:
                    pass

        result = dict(cached)
        # 현 종목 상대 위치
        fin_self = CACHE_DIR / f"fin_{ticker}_{date[:6]}.json"
        if fin_self.exists():
            try:
                fin = json.loads(fin_self.read_text(encoding="utf-8"))
                per = float(fin.get("per", 0) or 0)
                pbr = float(fin.get("pbr", 0) or 0)
                if per > 0 and result.get("sector_per_median"):
                    result["per_vs_sector"] = round(per / result["sector_per_median"] - 1, 2)
                if pbr > 0 and result.get("sector_pbr_median"):
                    result["pbr_vs_sector"] = round(pbr / result["sector_pbr_median"] - 1, 2)
            except Exception:
                pass
        return result

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
            # pykrx returns Korean column names — normalize to English
            col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
                       "거래량": "volume", "거래대금": "amount"}
            df.columns = [col_map.get(c, c.lower().replace(" ", "_")) for c in df.columns]
            return df
        except Exception as e:
            logger.debug(f"pykrx 지수 실패 ({index_code}): {e}")
            return pd.DataFrame()
