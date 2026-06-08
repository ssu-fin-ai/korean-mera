"""한국 주식 데이터 수집: pykrx (primary) + FinanceDataReader (fallback)

데이터 소스 우선순위:
  1. pykrx: KRX 공식 데이터 (일별 OHLCV, 시가총액, PER/PBR/DIV 등)
  2. FinanceDataReader: pykrx 실패 시 자동 대체 (네이버 금융 등에서 수집)
  3. OpenDartReader: DART 공시 데이터 (재무제표, 공시 목록)
  4. 네이버 금융 API: PER/PBR/투자자별 거래 (pykrx 실패 시 fallback)

모든 API 결과는 로컬 파일(Parquet/JSON)로 캐싱해 재요청을 최소화한다.
"""

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

# pykrx 지수코드 → FDR 심볼 매핑 (지수 fallback용)
_INDEX_FDR = {"1001": "KS11", "2001": "KQ11"}


class KoreanStockCollector:
    def __init__(self):
        self._dart = None
        # ticker → DART 8자리 법인코드 인스턴스 캐시 (API 중복 호출 방지)
        self._dart_code_map: dict[str, str] = {}

    @property
    def dart(self):
        """OpenDartReader 싱글톤 (DART_API_KEY 없으면 None 반환)"""
        if self._dart is None and DART_AVAILABLE and DART_API_KEY:
            self._dart = odr.OpenDartReader(DART_API_KEY)
        return self._dart

    @staticmethod
    def _dart_year_for_date(date: str) -> tuple[int, str]:
        """날짜 기준 가장 최근 제출된 DART 사업보고서 연도·보고서코드 반환.

        사업보고서 제출 기한: 3월 31일.
        4월 이후 → 전년도 사업보고서 사용 (이미 제출됨).
        1~3월 → 전전년도 사업보고서 사용 (전년도 미제출 상태).
        reprt_code "11011" = 사업보고서.
        """
        dt = datetime.strptime(date, "%Y%m%d")
        if dt.month >= 4:
            return dt.year - 1, "11011"
        return dt.year - 2, "11011"

    def _ticker_to_dart_code(self, ticker: str) -> str | None:
        """주식 티커(6자리) → DART 법인코드(8자리) 변환 (인스턴스 캐시 사용)

        DART API는 법인코드 기반이므로 티커를 먼저 변환해야 한다.
        corp_codes DataFrame에서 stock_code로 매칭한다.
        """
        if self.dart is None:
            return None
        if ticker in self._dart_code_map:
            return self._dart_code_map[ticker]
        try:
            codes_df = self.dart.corp_codes
            if codes_df is None or codes_df.empty:
                return None
            # 6자리로 zero-padding 후 비교
            mask = codes_df["stock_code"].astype(str).str.zfill(6) == ticker.zfill(6)
            rows = codes_df[mask]
            if not rows.empty:
                code = str(rows.iloc[0]["corp_code"])
                self._dart_code_map[ticker] = code  # 다음 호출을 위해 캐싱
                return code
        except Exception as e:
            logger.debug(f"DART 법인코드 조회 실패 ({ticker}): {e}")
        return None

    # ── 유니버스 ──────────────────────────────────────────────

    def get_universe(self) -> list[str]:
        """KOSPI200 + KOSDAQ150 종목 코드 반환 (pykrx 실패 시 FDR fallback)

        settings.yaml의 data.universe 설정에 따라 인덱스 선택.
        pykrx가 완전히 실패하면 FDR로 KOSPI 상위 200개 반환.
        """
        tickers = set()
        cfg = SETTINGS["data"]["universe"]

        # 1차: pykrx로 인덱스 구성 종목 조회
        if cfg.get("kospi200"):
            tickers.update(self._get_index_tickers("1028", "KOSPI200"))
        if cfg.get("kosdaq150"):
            tickers.update(self._get_index_tickers("2203", "KOSDAQ150"))

        # pykrx 전체 실패 시 FDR로 대체
        if not tickers:
            logger.warning("pykrx 유니버스 실패 → FinanceDataReader fallback")
            tickers.update(self._get_universe_fdr())

        result = sorted(tickers)
        logger.info(f"유니버스: {len(result)}개 종목")
        return result

    def _get_index_tickers(self, index_code: str, name: str) -> list[str]:
        """pykrx로 특정 인덱스 구성 종목 코드 반환"""
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
        """일별 OHLCV — pykrx 실패 시 FDR fallback, 로컬 파일 캐시 사용

        캐시 키: {ticker}_{start}_{end}.parquet
        날짜 범위가 바뀌면 다른 캐시 파일 사용 (재다운로드).
        """
        cache_path = CACHE_DIR / f"{ticker}_{start}_{end}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        # 1차 시도: pykrx (KRX 공식 데이터)
        df = self._get_ohlcv_pykrx(ticker, start, end)
        if df.empty:
            # pykrx 실패 시 FDR로 재시도
            df = self._get_ohlcv_fdr(ticker, start, end)

        if not df.empty:
            df["ticker"] = ticker
            df.to_parquet(cache_path)
        return df

    def _get_ohlcv_pykrx(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """pykrx로 OHLCV + 시가총액 수집, 컬럼 수 가변(5~7개)에 동적 대응"""
        try:
            df = stock.get_market_ohlcv(start, end, ticker)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            ncols = len(df.columns)
            # pykrx 버전에 따라 반환 컬럼 수가 다름 → 동적 처리
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
            # 시가총액 데이터 조인 (별도 API 호출)
            try:
                df_cap = stock.get_market_cap(start, end, ticker)
                df_cap.index = pd.to_datetime(df_cap.index)
                if "시가총액" in df_cap.columns:
                    df_cap = df_cap[["시가총액", "거래대금"]].rename(
                        columns={"시가총액": "mktcap", "거래대금": "turnover"}
                    )
                    df = df.join(df_cap, how="left")
            except Exception:
                pass  # 시가총액 실패는 치명적이지 않으므로 무시
            return df
        except Exception as e:
            logger.debug(f"pykrx OHLCV 실패 ({ticker}): {e}")
            return pd.DataFrame()

    def _get_ohlcv_fdr(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """FDR로 OHLCV 수집, 컬럼명을 pykrx 형식으로 정규화"""
        try:
            # FDR은 YYYY-MM-DD 형식 사용
            start_d = f"{start[:4]}-{start[4:6]}-{start[6:]}"
            end_d = f"{end[:4]}-{end[4:6]}-{end[6:]}"
            df = fdr.DataReader(ticker, start_d, end_d)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            # FDR 컬럼명 → 내부 표준 컬럼명으로 변환
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
        """여러 종목 OHLCV 순차 수집 (API 과부하 방지를 위해 delay 간격 적용)"""
        result = {}
        for i, ticker in enumerate(tickers):
            try:
                result[ticker] = self.get_ohlcv(ticker, start, end)
                if (i + 1) % 50 == 0:
                    logger.info(f"  {i+1}/{len(tickers)} 수집 완료")
            except Exception as e:
                logger.warning(f"{ticker} 수집 실패: {e}")
            time.sleep(delay)  # API 요청 간격 (서버 부하 방지)
        return result

    # ── 섹터 ─────────────────────────────────────────────────

    def get_sector_map(self) -> pd.DataFrame:
        """KRX 종목별 섹터·이름·시장 매핑 DataFrame (7일 캐시)

        FDR StockListing으로 KOSPI+KOSDAQ 전체 종목 메타데이터 수집.
        캐시 파일이 7일 이내면 재사용, 이후면 갱신.
        """
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
        """종목 섹터명 반환 (없으면 '기타')"""
        try:
            sector_map = self.get_sector_map()
            if ticker in sector_map.index:
                return str(sector_map.loc[ticker, "sector"])
        except Exception:
            pass
        return "기타"

    # ── DART 공시 ────────────────────────────────────────────

    def get_recent_filings(self, ticker: str, days: int = 30, ref_date: str = None) -> list[dict]:
        """최근 공시 목록 조회 (정기보고서 제외, 일별 캐시)

        조회 대상:
          B: 주요사항보고서 (수주·계약·자기주식 취득 등)
          D: 지분공시 (대주주 지분 변동)
          E: 기타 공시

        정기보고서(A)는 투자 신호와 무관하므로 제외.
        일별 캐시로 같은 날 중복 API 호출 방지.
        """
        import json
        if self.dart is None:
            return []
        ref = datetime.strptime(ref_date, "%Y%m%d") if ref_date else datetime.today()
        ref_str = ref.strftime("%Y%m%d")
        cache_path = CACHE_DIR / f"filings_{ticker}_{ref_str}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        corp_code = self._ticker_to_dart_code(ticker) or ticker
        start = (ref - timedelta(days=days)).strftime("%Y-%m-%d")
        end = ref.strftime("%Y-%m-%d")
        rows = []
        # B(주요사항), D(지분공시), E(기타) 각각 조회 후 합산
        for kind in ("B", "D", "E"):
            try:
                df = self.dart.list(corp_code, start=start, end=end, kind=kind, final="Y")
                if df is not None and not df.empty:
                    rows.append(df[["rcept_dt", "report_nm", "corp_name"]])
            except Exception as e:
                logger.debug(f"DART 공시 조회 실패 ({corp_code}, kind={kind}): {e}")

        result = []
        if rows:
            # 날짜 내림차순으로 정렬 후 최대 5개만 반환
            combined = pd.concat(rows).sort_values("rcept_dt", ascending=False)
            result = combined.head(5).to_dict("records")

        try:
            cache_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
        return result

    def get_financials(self, ticker: str, date: str) -> dict:
        """PER/PBR/배당수익률(pykrx) + 재무제표(DART) 통합 재무 데이터 (월별 캐시)

        수집 항목:
          pykrx: PER, PBR, EPS, BPS, DIV, DPS (역사적 날짜별)
          DART:  매출/이익 YoY, ROA/ROE, 부채비율, 유동비율, FCF, 이자보상배율
          네이버: pykrx 실패 시 현재 PER/PBR (백데이터 미지원)

        파생 지표:
          ROE: DART 우선, 없으면 EPS/BPS 근사 계산
          배당성향: DPS/EPS × 100
          Graham Number: √(22.5 × EPS × BPS)
        """
        import json
        # 월별 캐시 (같은 달 재호출 시 API 생략)
        cache_path = CACHE_DIR / f"fin_{ticker}_{date[:6]}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        result: dict = {}

        # pykrx: 역사적 날짜별 PER, PBR, EPS, BPS, DIV, DPS
        try:
            start_7d = (datetime.strptime(date, "%Y%m%d") - timedelta(days=7)).strftime("%Y%m%d")
            df = stock.get_market_fundamental(start_7d, date, ticker)
            if not df.empty:
                row = df.iloc[-1]  # 가장 최근 날짜 데이터 사용
                result.update({
                    "per": round(float(row.get("PER", 0) or 0), 2),
                    "pbr": round(float(row.get("PBR", 0) or 0), 2),
                    "eps": int(row.get("EPS", 0) or 0),
                    "bps": int(row.get("BPS", 0) or 0),
                    "div": round(float(row.get("DIV", 0) or 0), 2),  # 배당수익률(%)
                    "dps": int(row.get("DPS", 0) or 0),              # 주당배당금(원)
                })
        except Exception as e:
            logger.debug(f"pykrx fundamental 실패 ({ticker}): {e}")

        # pykrx 실패 시 네이버 금융으로 fallback (현재값만, 백데이터 미지원)
        if not result:
            try:
                result.update(self._get_naver_fundamental(ticker))
            except Exception as e:
                logger.debug(f"네이버 fundamental fallback 실패 ({ticker}): {e}")

        # DART 재무제표에서 매출/이익/ROA/ROE/유동비율/이자보상배율 수집
        dart_data = self.get_financial_summary(ticker, date)
        result.update(dart_data)

        # ROE 우선순위: DART 계산값 > EPS/BPS 근사값
        eps = result.get("eps", 0)
        bps = result.get("bps", 0)
        dps = result.get("dps", 0)
        if "roe_dart" in result:
            result["roe"] = result.pop("roe_dart")  # DART 계산 ROE를 최종 roe로 승격
        elif eps and bps and bps > 0:
            # EPS/BPS 근사: 당기순이익 / 자기자본 × 100
            result["roe"] = round(float(eps) / float(bps) * 100, 2)

        # 배당성향(%) = DPS / EPS × 100 (30~70%가 지속 가능한 적정 수준)
        if eps and dps and eps > 0:
            result["payout_ratio"] = round(float(dps) / float(eps) * 100, 1)

        # Graham Number: 내재가치 추정 공식 (벤저민 그레이엄)
        # 현재가 < Graham Number → 저평가 신호
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
        """네이버 금융 API에서 PER/PBR/EPS/BPS/배당수익률 조회 (주말 포함 상시 동작)

        pykrx는 KRX 거래일에만 데이터를 제공하지만
        네이버 금융 API는 항상 현재 시점 기준값을 반환한다.
        """
        import urllib.request as ur
        import re

        url = f"https://m.stock.naver.com/api/stock/{ticker}/integration"
        req = ur.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with ur.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())

        def _parse(val: str) -> float:
            """쉼표·특수문자 제거 후 float 변환"""
            cleaned = re.sub(r"[^\d.]", "", str(val).replace(",", ""))
            return float(cleaned) if cleaned else 0.0

        # 네이버 API code명 → 내부 키명 매핑
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
        """DART 재무제표에서 핵심 재무 지표 추출

        DART finstate_all()로 연결재무제표(IS/BS/CF) 파싱:
          IS (손익계산서): 매출, 영업이익, 순이익, 이자비용
          BS (재무상태표): 총자산, 총부채, 자기자본, 유동자산/부채
          CF (현금흐름표): 영업활동현금흐름(OCF), CAPEX

        파생 지표 계산:
          YoY 성장률 = (당기 - 전기) / |전기| × 100
          ROA = 순이익 / 총자산 × 100
          ROE = 순이익 / 자기자본 × 100
          부채비율 = 총부채 / 자기자본 × 100
          유동비율 = 유동자산 / 유동부채 × 100
          이자보상배율 = 영업이익 / 이자비용
          FCF = OCF - CAPEX
        """
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
                """쉼표 구분 숫자 문자열 → int 변환"""
                try:
                    return int(str(val).replace(",", "").strip())
                except Exception:
                    return 0

            def _yoy(curr: int, prev: int) -> float | None:
                """전기 대비 YoY 성장률(%) 계산, 전기 0이면 None"""
                if prev and prev != 0:
                    return round((curr - prev) / abs(prev) * 100, 1)
                return None

            metrics: dict = {}
            # 재무상태표 누적 변수
            _ni = _ni_prev = _ta = _te = _td = _ocf = _capex = 0
            _rev_prev = _opi_prev = _opi_num = 0
            _ca = _cl = _int_exp = 0

            for _, row in df.iterrows():
                acc = str(row.get("account_nm", ""))   # 계정명
                sj  = str(row.get("sj_div", ""))       # 재무제표 구분 (IS/BS/CF 등)
                raw  = str(row.get("thstrm_amount", "0"))  # 당기 금액
                num  = _to_int(raw)
                prev = _to_int(row.get("frmtrm_amount", "0"))  # 전기 금액

                # ── 손익계산서 (IS/CIS) ────────────────────────────────────
                if sj in ("IS", "CIS"):
                    if "매출" in acc and "원가" not in acc and "비용" not in acc:
                        metrics["revenue"] = raw
                        _rev_prev = prev
                    elif "영업이익" in acc and "손실" not in acc:
                        metrics["op_income"] = raw
                        _opi_prev = prev
                        _opi_num = num
                    elif "당기순이익" in acc and "비지배" not in acc and "net_income" not in metrics:
                        # 비지배지분 순이익 제외, 최초 발견된 것만 사용
                        metrics["net_income"] = raw
                        _ni, _ni_prev = num, prev
                    elif "이자비용" in acc and _int_exp == 0:
                        _int_exp = abs(num)  # 이자비용은 음수로 표시되는 경우도 있어 절댓값

                # ── 재무상태표 (BS) ────────────────────────────────────────
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

                # ── 현금흐름표 (CF) ────────────────────────────────────────
                elif sj == "CF":
                    if "영업활동" in acc and "현금" in acc and _ocf == 0:
                        _ocf = num  # 영업활동현금흐름
                    elif "유형자산" in acc and "취득" in acc:
                        _capex = num  # CAPEX: 유형자산 취득 지출 (음수=현금유출)

            # ── YoY 성장률 계산 ──────────────────────────────────────────
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

            # ── 수익성 지표 ──────────────────────────────────────────────
            if _ni and _ta > 0:
                metrics["roa"] = round(_ni / _ta * 100, 2)
            if _ni and _te > 0:
                metrics["roe_dart"] = round(_ni / _te * 100, 2)

            # ── 안정성 지표 ──────────────────────────────────────────────
            if _td and _te > 0:
                metrics["debt_ratio"] = round(_td / _te * 100, 1)   # 100% 이하 = 안정
            if _ca and _cl > 0:
                metrics["current_ratio"] = round(_ca / _cl * 100, 1)  # 150% 이상 = 양호
            if _opi_num and _int_exp > 0:
                metrics["interest_coverage"] = round(_opi_num / _int_exp, 1)  # 3배 이상 = 안전

            # ── 현금흐름 ─────────────────────────────────────────────────
            # FCF = 영업활동현금흐름 - CAPEX (DART는 CAPEX를 양수로 보고하는 경우도 있음)
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
        """5일 기관/외국인 순매수 수량 + 외국인 보유비율 (네이버 금융, 일별 캐시)

        dealTrendInfos: 최근 5일 투자자별 거래 데이터
        inst_net_5d: 5일 기관 순매수 합계 (양수 = 기관 매수 우위)
        foreign_net_5d: 5일 외국인 순매수 합계
        foreign_hold_ratio: 외국인 보유비율(%)
        """
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
                """쉼표 구분 정수 변환"""
                try:
                    return int(str(s).replace(",", "").strip())
                except Exception:
                    return 0

            deal = data.get("dealTrendInfos", [])
            if deal:
                # 5일치 기관/외국인 순매수 수량 합산
                result["inst_net_5d"] = sum(_parse_q(d.get("organPureBuyQuant", "0")) for d in deal)
                result["foreign_net_5d"] = sum(_parse_q(d.get("foreignerPureBuyQuant", "0")) for d in deal)
                try:
                    # 외국인 보유비율: 첫 번째 항목(최신일)에서 가져옴
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
        """공매도 비중(%) + 5일 평균 공매도 비중 (pykrx, 일별 캐시)

        short_ratio: 최근일 전체 거래량 대비 공매도 비중(%)
        short_ratio_5d_avg: 최근 5일 평균 (추세 파악용)
        높은 공매도 비중 = 기관/외국인의 하락 베팅 신호
        """
        cache_path = CACHE_DIR / f"short_{ticker}_{date}.json"
        if cache_path.exists():
            try:
                return json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        result: dict = {}
        try:
            # 최근 14일치 조회 후 5일 평균 계산
            start = (datetime.strptime(date, "%Y%m%d") - timedelta(days=14)).strftime("%Y%m%d")
            df = stock.get_shorting_volume_by_date(start, date, ticker)
            if not df.empty:
                # pykrx 버전에 따라 컬럼명이 다를 수 있으므로 "비중" 포함 컬럼 동적 탐색
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
        """섹터 동종 종목의 PER/PBR 중앙값 + 현 종목 상대 위치 계산 (월별 캐시)

        per_vs_sector: 현 종목 PER / 섹터 중앙값 PER - 1
          음수 = 섹터 대비 저PER (저평가 신호)
        pbr_vs_sector: 현 종목 PBR / 섹터 중앙값 PBR - 1
          음수 = 섹터 대비 저PBR (저평가 신호)

        동종 종목의 fin_ 캐시 파일에서 읽어 API 호출 없이 계산.
        """
        sector_map = self.get_sector_map()
        if ticker not in sector_map.index:
            return {}

        sector = str(sector_map.loc[ticker, "sector"])
        if not sector or sector in ("기타", "nan"):
            return {}

        # 섹터명을 파일명에 사용할 수 있도록 특수문자 치환
        cache_key = sector.replace(" ", "_").replace("/", "_")
        cache_path = CACHE_DIR / f"sector_avg_{cache_key}_{date[:6]}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            # 같은 섹터 종목의 기존 fin_ 캐시 파일에서 PER/PBR 수집
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
                    # 이상치 제거: PER 0~200, PBR 0~50 범위만 포함
                    if 0 < per < 200:
                        pers.append(per)
                    if 0 < pbr < 50:
                        pbrs.append(pbr)
                except Exception:
                    pass

            cached = {}
            # 최소 3개 이상 샘플이 있어야 의미 있는 중앙값
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
        # 현 종목 상대 위치 계산 (섹터 중앙값 대비 비율 - 1)
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
        """지수 OHLCV — pykrx 실패 시 FDR fallback

        1001: KOSPI, 2001: KOSDAQ
        KOSPI 지수는 KOSPI 대비 상대수익률·베타 계산에 사용된다.
        """
        df = self._get_index_pykrx(index_code, start, end)
        if df.empty:
            fdr_symbol = _INDEX_FDR.get(index_code, "KS11")
            df = self._get_ohlcv_fdr(fdr_symbol, start, end)
            logger.info(f"지수 {index_code} → FDR({fdr_symbol}) fallback")
        return df

    def _get_index_pykrx(self, index_code: str, start: str, end: str) -> pd.DataFrame:
        """pykrx로 지수 OHLCV 수집, 한글 컬럼명을 영문으로 정규화"""
        try:
            df = stock.get_index_ohlcv(start, end, index_code)
            if df.empty:
                return pd.DataFrame()
            df.index = pd.to_datetime(df.index)
            # pykrx는 한글 컬럼명 반환 → 내부 표준 영문명으로 변환
            col_map = {"시가": "open", "고가": "high", "저가": "low", "종가": "close",
                       "거래량": "volume", "거래대금": "amount"}
            df.columns = [col_map.get(c, c.lower().replace(" ", "_")) for c in df.columns]
            return df
        except Exception as e:
            logger.debug(f"pykrx 지수 실패 ({index_code}): {e}")
            return pd.DataFrame()
