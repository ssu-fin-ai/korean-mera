"""기술적 지표 계산 및 정규화

OHLCV DataFrame에 RSI·MACD·볼린저밴드·ADX·OBV 등 기술적 지표를 추가하고
특정 날짜 기준 스냅샷 벡터를 생성한다.

pandas_ta 라이브러리를 활용해 지표를 계산하며,
dropna는 label_ 컬럼(미래 수익률)을 제외하고 수행해
최근 20일 행이 소실되지 않도록 한다.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger


class FeatureEngineer:
    def __init__(self, window: int = 20):
        # window: 스냅샷 벡터 생성 시 사용할 최근 N일 수익률 시계열 길이
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV DataFrame → 전체 기술적 지표 추가

        처리 순서: 수익률 → 추세 → 모멘텀 → 변동성 → 거래량
        label_ 컬럼은 dropna 대상에서 제외 (최근 20일 행 보존).
        """
        df = df.copy()
        df = self._add_returns(df)
        df = self._add_trend(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume(df)
        # label_ 컬럼만 제외하고 나머지 피처의 NaN 행 제거
        feature_cols = [c for c in df.columns if not c.startswith("label_")]
        return df.dropna(subset=feature_cols)

    # ── 수익률 ────────────────────────────────────────────────

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """과거 수익률(ret_Nd) + 미래 레이블(label_Nd) 추가

        label_5d/10d/20d는 N일 후 수익률로, 백테스트와 RAG 패턴 평가에 사용.
        shift(-N)으로 미래 값을 현재 행에 대응시킨다.
        """
        df["ret_1d"] = df["close"].pct_change()
        df["ret_3d"] = df["close"].pct_change(3)
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_10d"] = df["close"].pct_change(10)
        df["ret_20d"] = df["close"].pct_change(20)
        df["ret_60d"] = df["close"].pct_change(60)
        # 향후 레이블: 현재 행에서 N일 후 종가 수익률 (백테스트/RAG 정답 데이터)
        df["label_5d"] = df["close"].pct_change(5).shift(-5)
        df["label_10d"] = df["close"].pct_change(10).shift(-10)
        df["label_20d"] = df["close"].pct_change(20).shift(-20)
        return df

    # ── 추세 ──────────────────────────────────────────────────

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """이동평균·MACD·ADX 추가

        MA5/20/60/120 및 각 대비 현재가 괴리율(close_to_maN)을 계산.
        MACD 히스토그램(macd_diff) 양수 = 골든크로스 상태.
        ADX 25 이상 = 뚜렷한 방향성 추세.
        """
        for w in [5, 20, 60, 120]:
            df[f"ma{w}"] = df["close"].rolling(w).mean()
            # 현재가가 이동평균 대비 몇 % 위/아래인지 (양수=위)
            df[f"close_to_ma{w}"] = df["close"] / df[f"ma{w}"] - 1

        macd = ta.macd(df["close"])
        if macd is not None and not macd.empty:
            # pandas_ta 컬럼 순서: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            df["macd"] = macd.iloc[:, 0]          # MACD 선 (EMA12 - EMA26)
            df["macd_signal"] = macd.iloc[:, 2]   # 시그널 선 (EMA9 of MACD)
            df["macd_diff"] = df["macd"] - df["macd_signal"]  # 히스토그램 (양수=골든크로스)

        adx = ta.adx(df["high"], df["low"], df["close"])
        if adx is not None and not adx.empty:
            df["adx"] = adx.iloc[:, 0]  # ADX: 추세 강도 (방향 무관)

        return df

    # ── 모멘텀 ────────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """RSI·Stochastic·CCI·Williams %R 추가

        RSI 14: 70+ 과매수, 30- 과매도 기준.
        Stochastic: K(빠름), D(느림) 교차로 매매 신호 포착.
        """
        df["rsi"] = ta.rsi(df["close"], length=14)

        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]  # %K 빠른 선
            df["stoch_d"] = stoch.iloc[:, 1]  # %D 느린 선 (신호선)

        df["cci"] = ta.cci(df["high"], df["low"], df["close"])
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"])
        return df

    # ── 변동성 ────────────────────────────────────────────────

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """볼린저밴드·ATR·역사적변동성(20일 연환산) 추가

        bb_pct: 0=하단, 0.5=중간, 1=상단 (현재가의 밴드 내 위치).
        bb_width: 밴드 폭 (변동성 확장/수축 측정).
        hist_vol_20: 20일 일간 수익률 표준편차 × √252 (연환산 변동성).
        """
        bb = ta.bbands(df["close"])
        if bb is not None and not bb.empty:
            df["bb_lower"] = bb.iloc[:, 0]  # 하단 밴드 (μ - 2σ)
            df["bb_mid"] = bb.iloc[:, 1]    # 중간 밴드 (20일 이동평균)
            df["bb_upper"] = bb.iloc[:, 2]  # 상단 밴드 (μ + 2σ)
            # 1e-9: 밴드 폭이 0일 때 division by zero 방지
            df["bb_pct"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"] + 1e-9
            )
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)

        df["atr"] = ta.atr(df["high"], df["low"], df["close"])  # 평균 실제 범위
        # 연환산 역사적 변동성: 20일 표준편차 × √252 거래일
        df["hist_vol_20"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
        return df

    # ── 거래량 ────────────────────────────────────────────────

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """거래량 비율·OBV·MFI·VWAP 추가

        volume_ratio: 오늘 거래량 / 20일 평균 (1.5+ = 평균보다 50% 많음).
        OBV: 누적 거래량으로 추세 확인.
        MFI: 거래량 가중 RSI (자금 유출입 강도).
        VWAP: 거래대금 / 거래량 = 거래량 가중 평균 가격.
        """
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        # +1: 거래량이 0인 날(공휴일 등) division by zero 방지
        df["volume_ratio"] = df["volume"] / (df["vol_ma20"] + 1)

        df["obv"] = ta.obv(df["close"], df["volume"])
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"])

        # amount 컬럼이 있을 때만 VWAP 계산 (pykrx 수집 시 제공)
        if "amount" in df.columns:
            df["vwap"] = df["amount"] / (df["volume"] + 1)

        return df

    # ── 상대 강도 (시장/섹터 대비) ────────────────────────────

    def get_52w_position(self, df: pd.DataFrame, date: str) -> dict:
        """52주 고점/저점 대비 현재가 위치 + 현재가 반환

        date 이전 252 거래일(약 1년)의 고/저가를 기준으로
        pct_from_52w_high (음수 = 고점 대비 하락률),
        pct_from_52w_low  (양수 = 저점 대비 상승률)을 계산한다.
        """
        available = df.index.strftime("%Y-%m-%d")
        valid = available[available <= date]
        if valid.empty:
            return {}
        actual_date = valid[-1]  # 요청 날짜 이전 가장 가까운 거래일
        row = df[df.index.strftime("%Y-%m-%d") == actual_date].iloc[-1]
        close = float(row.get("close", 0))
        if close <= 0:
            return {}
        # 해당 날짜 기준 최근 252 거래일
        window_52w = df[df.index.strftime("%Y-%m-%d") <= actual_date].tail(252)
        result = {"current_price": int(close)}
        if not window_52w.empty:
            high_52w = float(window_52w["high"].max())
            low_52w  = float(window_52w["low"].min())
            if high_52w > 0:
                result["pct_from_52w_high"] = round(close / high_52w - 1, 4)  # 음수: 고점 대비 하락
            if low_52w > 0:
                result["pct_from_52w_low"]  = round(close / low_52w  - 1, 4)  # 양수: 저점 대비 상승
        return result

    def add_relative_strength(self, df: pd.DataFrame,
                               market_df: pd.DataFrame) -> pd.DataFrame:
        """KOSPI 지수 대비 상대수익률·베타 추가

        rel_strength_5d/20d: 개별 종목 수익률 - KOSPI 수익률 (양수 = 시장 대비 아웃퍼폼).
        beta_20d: 20일 롤링 공분산 / 시장 분산 (1=시장과 동일 변동성).
        """
        market_ret = market_df["close"].pct_change().reindex(df.index)
        # 종목 5/20일 수익률 - KOSPI 같은 기간 수익률 합계
        df["rel_strength_5d"] = df["ret_5d"] - market_ret.rolling(5).sum()
        df["rel_strength_20d"] = df["ret_20d"] - market_ret.rolling(20).sum()
        # 베타 = Cov(종목 일간수익률, 시장 일간수익률) / Var(시장 일간수익률)
        df["beta_20d"] = (
            df["ret_1d"].rolling(20).cov(market_ret)
            / market_ret.rolling(20).var()
        )
        return df

    # ── 스냅샷 (패턴 벡터) ───────────────────────────────────

    def get_snapshot_vector(self, df: pd.DataFrame, date: str) -> dict | None:
        """특정 날짜 기준 패턴 수치 벡터 반환

        해당 날짜의 거래가 없으면 가장 가까운 이전 거래일을 사용.
        returns_series: 최근 window일의 일간 수익률 시계열 (임베딩 텍스트 생성용).
        나머지는 스칼라 지표 (screener 스코어링·전문가 포맷터에서 사용).
        """
        available = df.index.strftime("%Y-%m-%d")
        valid = available[available <= date]
        if valid.empty:
            return None
        actual_date = valid[-1]  # 요청 날짜 이전 마지막 거래일

        row = df[df.index.strftime("%Y-%m-%d") == actual_date].iloc[-1]
        # 최근 window일 데이터 (수익률 시계열 추출용)
        window_data = df[df.index.strftime("%Y-%m-%d") <= actual_date].tail(self.window)

        return {
            # 최근 N일 일간 수익률 시계열 (NaN → 0으로 채움)
            "returns_series": window_data["ret_1d"].fillna(0).tolist(),
            # 스칼라 지표 (screener 스코어링 + 전문가 포맷터에서 활용)
            "ret_1d": float(row.get("ret_1d", 0)),       # 전일 대비 수익률
            "ret_5d": float(row.get("ret_5d", 0)),        # 5일 수익률
            "ret_20d": float(row.get("ret_20d", 0)),      # 20일 수익률
            "rsi": float(row.get("rsi", 50)),             # RSI (14일)
            "macd_diff": float(row.get("macd_diff", 0)),  # MACD 히스토그램
            "bb_pct": float(row.get("bb_pct", 0.5)),      # 볼린저밴드 위치 (0~1)
            "volume_ratio": float(row.get("volume_ratio", 1)),  # 거래량/20일평균
            "hist_vol_20": float(row.get("hist_vol_20", 0)),    # 역사적변동성(연환산)
            "close_to_ma20": float(row.get("close_to_ma20", 0)),  # MA20 대비 괴리율
            "close_to_ma60": float(row.get("close_to_ma60", 0)),  # MA60 대비 괴리율
            "adx": float(row.get("adx", 20)),             # 추세 강도
            "mfi": float(row.get("mfi", 50)),             # 자금 흐름 지수
            "beta_20d": float(row.get("beta_20d", 1)),    # KOSPI 대비 베타
        }
