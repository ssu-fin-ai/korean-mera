"""기술적 지표 계산 및 정규화"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger


class FeatureEngineer:
    def __init__(self, window: int = 20):
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """OHLCV DataFrame → 전체 피처 추가"""
        df = df.copy()
        df = self._add_returns(df)
        df = self._add_trend(df)
        df = self._add_momentum(df)
        df = self._add_volatility(df)
        df = self._add_volume(df)
        feature_cols = [c for c in df.columns if not c.startswith("label_")]
        return df.dropna(subset=feature_cols)

    # ── 수익률 ────────────────────────────────────────────────

    def _add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ret_1d"] = df["close"].pct_change()
        df["ret_3d"] = df["close"].pct_change(3)
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_10d"] = df["close"].pct_change(10)
        df["ret_20d"] = df["close"].pct_change(20)
        df["ret_60d"] = df["close"].pct_change(60)
        # 향후 레이블 (백테스트용)
        df["label_5d"] = df["close"].pct_change(5).shift(-5)
        df["label_10d"] = df["close"].pct_change(10).shift(-10)
        df["label_20d"] = df["close"].pct_change(20).shift(-20)
        return df

    # ── 추세 ──────────────────────────────────────────────────

    def _add_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in [5, 20, 60, 120]:
            df[f"ma{w}"] = df["close"].rolling(w).mean()
            df[f"close_to_ma{w}"] = df["close"] / df[f"ma{w}"] - 1

        macd = ta.macd(df["close"])
        if macd is not None and not macd.empty:
            df["macd"] = macd.iloc[:, 0]
            df["macd_signal"] = macd.iloc[:, 1]
            df["macd_diff"] = macd.iloc[:, 2]

        adx = ta.adx(df["high"], df["low"], df["close"])
        if adx is not None and not adx.empty:
            df["adx"] = adx.iloc[:, 0]

        return df

    # ── 모멘텀 ────────────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        df["rsi"] = ta.rsi(df["close"], length=14)

        stoch = ta.stoch(df["high"], df["low"], df["close"])
        if stoch is not None and not stoch.empty:
            df["stoch_k"] = stoch.iloc[:, 0]
            df["stoch_d"] = stoch.iloc[:, 1]

        df["cci"] = ta.cci(df["high"], df["low"], df["close"])
        df["williams_r"] = ta.willr(df["high"], df["low"], df["close"])
        return df

    # ── 변동성 ────────────────────────────────────────────────

    def _add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        bb = ta.bbands(df["close"])
        if bb is not None and not bb.empty:
            df["bb_lower"] = bb.iloc[:, 0]
            df["bb_mid"] = bb.iloc[:, 1]
            df["bb_upper"] = bb.iloc[:, 2]
            df["bb_pct"] = (df["close"] - df["bb_lower"]) / (
                df["bb_upper"] - df["bb_lower"] + 1e-9
            )
            df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (df["bb_mid"] + 1e-9)

        df["atr"] = ta.atr(df["high"], df["low"], df["close"])
        df["hist_vol_20"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
        return df

    # ── 거래량 ────────────────────────────────────────────────

    def _add_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["volume_ratio"] = df["volume"] / (df["vol_ma20"] + 1)

        df["obv"] = ta.obv(df["close"], df["volume"])
        df["mfi"] = ta.mfi(df["high"], df["low"], df["close"], df["volume"])

        if "amount" in df.columns:
            df["vwap"] = df["amount"] / (df["volume"] + 1)

        return df

    # ── 상대 강도 (시장/섹터 대비) ────────────────────────────

    def add_relative_strength(self, df: pd.DataFrame,
                               market_df: pd.DataFrame) -> pd.DataFrame:
        """KOSPI 대비 상대수익률 추가"""
        market_ret = market_df["close"].pct_change().reindex(df.index)
        df["rel_strength_5d"] = df["ret_5d"] - market_ret.rolling(5).sum()
        df["rel_strength_20d"] = df["ret_20d"] - market_ret.rolling(20).sum()
        df["beta_20d"] = (
            df["ret_1d"].rolling(20).cov(market_ret)
            / market_ret.rolling(20).var()
        )
        return df

    # ── 스냅샷 (패턴 벡터) ───────────────────────────────────

    def get_snapshot_vector(self, df: pd.DataFrame, date: str) -> dict | None:
        """특정 날짜 기준 패턴 수치 벡터 반환 (당일 없으면 가장 가까운 이전 거래일 사용)"""
        available = df.index.strftime("%Y-%m-%d")
        valid = available[available <= date]
        if valid.empty:
            return None
        actual_date = valid[-1]

        row = df[df.index.strftime("%Y-%m-%d") == actual_date].iloc[-1]
        window_data = df[df.index.strftime("%Y-%m-%d") <= actual_date].tail(self.window)

        return {
            # 수익률 시계열 (정규화)
            "returns_series": window_data["ret_1d"].fillna(0).tolist(),
            # 스칼라 피처
            "ret_5d": float(row.get("ret_5d", 0)),
            "ret_20d": float(row.get("ret_20d", 0)),
            "rsi": float(row.get("rsi", 50)),
            "macd_diff": float(row.get("macd_diff", 0)),
            "bb_pct": float(row.get("bb_pct", 0.5)),
            "volume_ratio": float(row.get("volume_ratio", 1)),
            "hist_vol_20": float(row.get("hist_vol_20", 0)),
            "close_to_ma20": float(row.get("close_to_ma20", 0)),
            "close_to_ma60": float(row.get("close_to_ma60", 0)),
            "adx": float(row.get("adx", 20)),
            "mfi": float(row.get("mfi", 50)),
        }
