"""신호 검증: 과거 포트폴리오 CSV로 수익률 계산"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from loguru import logger

matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from config import ROOT

REPORTS_DIR = ROOT / "reports"


def load_portfolios(reports_dir: Path = None) -> pd.DataFrame:
    """저장된 portfolio_*.csv 파일을 모두 로드"""
    reports_dir = reports_dir or REPORTS_DIR
    files = sorted(reports_dir.glob("portfolio_*.csv"))
    if not files:
        logger.warning("포트폴리오 CSV 없음")
        return pd.DataFrame()

    dfs = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def compute_returns(
    portfolio_df: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    hold_days: int = 5,
) -> pd.DataFrame:
    """각 포트폴리오 신호의 실제 수익률 계산"""
    rows = []
    for _, row in portfolio_df.iterrows():
        ticker = str(row["ticker"])
        date = str(row["date"])
        signal = row.get("signal", "BUY")
        confidence = float(row.get("confidence", 0.5))

        if ticker not in price_data:
            continue

        df = price_data[ticker]
        df.index = pd.to_datetime(df.index)
        idx = df.index.get_indexer([pd.to_datetime(date)], method="nearest")[0]

        if idx < 0 or idx + hold_days >= len(df):
            continue

        entry_price = df["close"].iloc[idx]
        exit_price = df["close"].iloc[idx + hold_days]
        actual_return = (exit_price - entry_price) / entry_price

        rows.append({
            "ticker": ticker,
            "name": row.get("name", ""),
            "sector": row.get("sector", ""),
            "date": date,
            "signal": signal,
            "confidence": confidence,
            "predicted_return": float(row.get("target_return", 0)),
            "actual_return": actual_return,
            "hit": int(actual_return > 0),
        })

    return pd.DataFrame(rows)


def _compute_arr(cum_series: pd.Series, periods_per_year: float) -> float:
    """누적 수익률 시계열 → 연환산 수익률 (ARR)"""
    n = len(cum_series)
    if n == 0:
        return 0.0
    total = float(cum_series.iloc[-1])
    return float(total ** (periods_per_year / n) - 1)


def _compute_mdd(cum_series: pd.Series) -> float:
    """누적 수익률 시계열 → 최대 낙폭 (MDD, 음수)"""
    if cum_series.empty:
        return 0.0
    peak = cum_series.cummax()
    drawdown = (cum_series - peak) / peak
    return float(drawdown.min())


def _compute_sortino(returns: pd.Series, periods_per_year: float) -> float:
    """연환산 Sortino ratio (하방 편차 기준)"""
    mean_r = returns.mean()
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    down_std = float(downside.std())
    if down_std == 0:
        return 0.0
    return float(mean_r / down_std * (periods_per_year ** 0.5))


def compute_metrics(result_df: pd.DataFrame, hold_days: int = 5) -> dict:
    """성과 지표 계산 (ARR, MDD, Sortino, Calmar 포함)"""
    if result_df.empty:
        return {}

    buy_df = result_df[result_df["signal"] == "BUY"].copy()
    if buy_df.empty:
        return {}

    buy_df = buy_df.sort_values("date")
    returns = buy_df["actual_return"]
    periods_per_year = 252 / hold_days

    cum = (1 + returns).cumprod()
    mean_r = float(returns.mean())
    std_r = float(returns.std()) if len(returns) > 1 else 0.0

    arr = _compute_arr(cum, periods_per_year)
    mdd = _compute_mdd(cum)
    sharpe = float(mean_r / std_r * (periods_per_year ** 0.5)) if std_r > 0 else 0.0
    sortino = _compute_sortino(returns, periods_per_year)
    calmar = float(arr / abs(mdd)) if mdd != 0 else 0.0

    metrics = {
        "total_signals": len(buy_df),
        "hit_rate": float(buy_df["hit"].mean()),
        "avg_return": mean_r,
        "median_return": float(returns.median()),
        "std_return": std_r,
        "sharpe": sharpe,
        "arr": arr,
        "mdd": mdd,
        "sortino": sortino,
        "calmar": calmar,
        "max_return": float(returns.max()),
        "min_return": float(returns.min()),
        "by_sector": buy_df.groupby("sector")["actual_return"].mean().to_dict(),
    }

    logger.info("=== 백테스트 결과 ===")
    logger.info(f"총 신호: {metrics['total_signals']}건")
    logger.info(f"적중률: {metrics['hit_rate']:.1%}")
    logger.info(f"평균 수익률: {mean_r:+.2%}")
    logger.info(f"ARR(연환산): {arr:+.1%}")
    logger.info(f"MDD: {mdd:.1%}")
    logger.info(f"샤프(연환산): {sharpe:.2f}")
    logger.info(f"소르티노: {sortino:.2f}")
    logger.info(f"칼마: {calmar:.2f}")
    return metrics


def plot_results(result_df: pd.DataFrame, save_path: Path = None, hold_days: int = 5):
    """수익률 분포, 누적 수익률, MDD, 섹터/신뢰도 시각화 (2×3)"""
    if result_df.empty:
        return

    buy_df = result_df[result_df["signal"] == "BUY"].copy()
    buy_df = buy_df.sort_values("date")
    returns = buy_df["actual_return"]
    periods_per_year = 252 / hold_days

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak

    arr = _compute_arr(cum, periods_per_year)
    mdd = _compute_mdd(cum)
    sortino = _compute_sortino(returns, periods_per_year)
    mean_r = float(returns.mean())
    std_r = float(returns.std()) if len(returns) > 1 else 0.0
    sharpe = float(mean_r / std_r * (periods_per_year ** 0.5)) if std_r > 0 else 0.0
    calmar = float(arr / abs(mdd)) if mdd != 0 else 0.0

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("MERA 한국주식 백테스트 결과", fontsize=14)
    x = range(len(buy_df))

    # [0,0] 수익률 분포
    axes[0, 0].hist(returns * 100, bins=50, color="steelblue", edgecolor="white")
    axes[0, 0].axvline(0, color="red", linestyle="--")
    axes[0, 0].set_title("수익률 분포 (%)")
    axes[0, 0].set_xlabel("수익률 (%)")

    # [0,1] 누적 수익률
    axes[0, 1].plot(x, cum.values, color="green", label="누적 수익")
    axes[0, 1].axhline(1, color="red", linestyle="--")
    axes[0, 1].set_title(f"누적 수익률 (ARR {arr:+.1%})")
    axes[0, 1].set_ylabel("누적 배수")
    axes[0, 1].legend()

    # [0,2] MDD 언더워터
    axes[0, 2].fill_between(x, drawdown.values * 100, 0, color="tomato", alpha=0.6)
    axes[0, 2].plot(x, drawdown.values * 100, color="darkred", linewidth=0.8)
    axes[0, 2].set_title(f"낙폭 (MDD {mdd:.1%})")
    axes[0, 2].set_ylabel("낙폭 (%)")

    # [1,0] 섹터별 평균 수익률
    sector_ret = buy_df.groupby("sector")["actual_return"].mean().sort_values()
    colors = ["tomato" if v < 0 else "steelblue" for v in sector_ret.values]
    axes[1, 0].barh(sector_ret.index, sector_ret.values * 100, color=colors)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_title("섹터별 평균 수익률 (%)")

    # [1,1] 신뢰도 vs 실제 수익률
    axes[1, 1].scatter(buy_df["confidence"], returns * 100,
                       alpha=0.4, color="purple", s=20)
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set_title("신뢰도 vs 실제 수익률")
    axes[1, 1].set_xlabel("신뢰도")
    axes[1, 1].set_ylabel("실제 수익률 (%)")

    # [1,2] 핵심 지표 요약
    labels = ["ARR", "MDD", "샤프", "소르티노", "칼마"]
    values = [arr, mdd, sharpe, sortino, calmar]
    bar_colors = ["green" if v >= 0 else "tomato" for v in values]
    bars = axes[1, 2].bar(labels, values, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, values):
        axes[1, 2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.01 if val >= 0 else -0.03),
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=9,
        )
    axes[1, 2].axhline(0, color="black", linewidth=0.5)
    axes[1, 2].set_title("핵심 리스크 지표")

    plt.tight_layout()
    save_path = save_path or REPORTS_DIR / "backtest_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"차트 저장: {save_path}")
    plt.close()
