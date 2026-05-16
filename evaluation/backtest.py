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


def compute_metrics(result_df: pd.DataFrame) -> dict:
    """성과 지표 계산"""
    if result_df.empty:
        return {}

    buy_df = result_df[result_df["signal"] == "BUY"]
    if buy_df.empty:
        return {}

    metrics = {
        "total_signals": len(buy_df),
        "hit_rate": buy_df["hit"].mean(),
        "avg_return": buy_df["actual_return"].mean(),
        "median_return": buy_df["actual_return"].median(),
        "std_return": buy_df["actual_return"].std(),
        "sharpe": (
            buy_df["actual_return"].mean() / buy_df["actual_return"].std()
            if buy_df["actual_return"].std() > 0 else 0
        ),
        "max_return": buy_df["actual_return"].max(),
        "min_return": buy_df["actual_return"].min(),
        "by_sector": buy_df.groupby("sector")["actual_return"].mean().to_dict(),
    }

    logger.info(f"=== 백테스트 결과 ===")
    logger.info(f"총 신호: {metrics['total_signals']}건")
    logger.info(f"적중률: {metrics['hit_rate']:.1%}")
    logger.info(f"평균 수익률: {metrics['avg_return']:+.2%}")
    logger.info(f"샤프: {metrics['sharpe']:.2f}")
    return metrics


def plot_results(result_df: pd.DataFrame, save_path: Path = None):
    """수익률 분포 및 누적 수익률 시각화"""
    if result_df.empty:
        return

    buy_df = result_df[result_df["signal"] == "BUY"].copy()
    buy_df = buy_df.sort_values("date")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("MERA 한국주식 백테스트 결과", fontsize=14)

    # 수익률 분포
    axes[0, 0].hist(buy_df["actual_return"] * 100, bins=50, color="steelblue", edgecolor="white")
    axes[0, 0].axvline(0, color="red", linestyle="--")
    axes[0, 0].set_title("수익률 분포 (%)")
    axes[0, 0].set_xlabel("수익률 (%)")

    # 누적 수익률
    cum_ret = (1 + buy_df["actual_return"]).cumprod()
    axes[0, 1].plot(range(len(cum_ret)), cum_ret.values, color="green")
    axes[0, 1].axhline(1, color="red", linestyle="--")
    axes[0, 1].set_title("누적 수익률")
    axes[0, 1].set_ylabel("누적 배수")

    # 섹터별 평균 수익률
    sector_ret = buy_df.groupby("sector")["actual_return"].mean().sort_values()
    colors = ["tomato" if v < 0 else "steelblue" for v in sector_ret.values]
    axes[1, 0].barh(sector_ret.index, sector_ret.values * 100, color=colors)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_title("섹터별 평균 수익률 (%)")

    # 신뢰도 vs 실제 수익률
    axes[1, 1].scatter(buy_df["confidence"], buy_df["actual_return"] * 100,
                       alpha=0.4, color="purple", s=20)
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set_title("신뢰도 vs 실제 수익률")
    axes[1, 1].set_xlabel("신뢰도")
    axes[1, 1].set_ylabel("실제 수익률 (%)")

    plt.tight_layout()
    save_path = save_path or REPORTS_DIR / "backtest_result.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"차트 저장: {save_path}")
    plt.close()
