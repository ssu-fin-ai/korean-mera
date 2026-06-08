"""신호 검증: 과거 포트폴리오 CSV로 수익률 계산

reports/ 폴더의 portfolio_*.csv를 로드해
각 BUY 신호에 대해 N일 후 실제 수익률을 계산하고
성과 지표(ARR·MDD·샤프·소르티노·칼마)와 시각화 차트를 생성한다.

--mode backtest 실행 시 이 모듈이 호출된다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from loguru import logger

# 한글 폰트 설정 (Windows: 맑은 고딕, 마이너스 기호 깨짐 방지)
matplotlib.rcParams["font.family"] = "Malgun Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False

from config import ROOT

REPORTS_DIR = ROOT / "reports"


def load_portfolios(reports_dir: Path = None) -> pd.DataFrame:
    """reports/ 폴더의 portfolio_*.csv 파일을 모두 로드해 하나의 DataFrame으로 합침

    파일이 없으면 경고 로그 후 빈 DataFrame 반환.
    """
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
    """각 포트폴리오 신호의 실제 수익률 계산

    편입일(date)로부터 hold_days 거래일 후의 종가 수익률을 계산한다.
    가장 가까운 날짜 인덱스를 nearest 방식으로 찾아 공휴일 처리.

    반환 컬럼: ticker, name, sector, date, signal, confidence,
               predicted_return, actual_return, hit(1=양수수익)
    """
    rows = []
    for _, row in portfolio_df.iterrows():
        ticker = str(row["ticker"])
        date = str(row["date"])
        signal = row.get("signal", "BUY")
        confidence = float(row.get("confidence", 0.5))

        if ticker not in price_data:
            continue  # 가격 데이터 없는 종목 스킵

        df = price_data[ticker]
        df.index = pd.to_datetime(df.index)
        # 편입일과 가장 가까운 거래일 인덱스 탐색 (공휴일 대응)
        idx = df.index.get_indexer([pd.to_datetime(date)], method="nearest")[0]

        if idx < 0 or idx + hold_days >= len(df):
            continue  # 데이터 범위 초과 시 스킵

        entry_price = df["close"].iloc[idx]          # 편입일 종가 (매수가)
        exit_price = df["close"].iloc[idx + hold_days]  # N일 후 종가 (매도가)
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
            "hit": int(actual_return > 0),  # 수익 = 1, 손실 = 0 (적중률 계산용)
        })

    return pd.DataFrame(rows)


def _compute_arr(cum_series: pd.Series, periods_per_year: float) -> float:
    """누적 수익률 시계열 → 연환산 수익률 (ARR)

    ARR = 마지막 누적값^(periods_per_year/N) - 1
    N이 클수록 더 많은 거래 회전이 있었음을 의미.
    """
    n = len(cum_series)
    if n == 0:
        return 0.0
    total = float(cum_series.iloc[-1])
    return float(total ** (periods_per_year / n) - 1)


def _compute_mdd(cum_series: pd.Series) -> float:
    """누적 수익률 시계열 → 최대 낙폭 (MDD, 음수)

    MDD: 특정 기간 동안 고점에서 저점까지의 최대 하락률.
    전략의 최악 시나리오 위험을 측정한다.
    """
    if cum_series.empty:
        return 0.0
    peak = cum_series.cummax()  # 현재까지의 최고값
    drawdown = (cum_series - peak) / peak  # 고점 대비 낙폭 비율
    return float(drawdown.min())  # 최대 낙폭 (음수)


def _compute_sortino(returns: pd.Series, periods_per_year: float) -> float:
    """연환산 Sortino ratio (하방 편차 기준 위험 조정 수익률)

    샤프와 달리 하방 편차(손실 구간의 변동성)만 사용해
    수익에 유리한 방향의 변동성을 위험으로 보지 않는다.
    """
    mean_r = returns.mean()
    downside = returns[returns < 0]  # 손실 거래만 선택
    if len(downside) < 2:
        return 0.0
    down_std = float(downside.std())
    if down_std == 0:
        return 0.0
    return float(mean_r / down_std * (periods_per_year ** 0.5))


def compute_metrics(result_df: pd.DataFrame, hold_days: int = 5) -> dict:
    """BUY 신호 기준 전체 성과 지표 계산

    반환 지표:
      total_signals:  총 BUY 신호 수
      hit_rate:       적중률 (양수 수익 비율)
      avg_return:     평균 수익률
      median_return:  중앙값 수익률 (이상치 영향 적음)
      std_return:     수익률 표준편차
      sharpe:         연환산 샤프 비율
      arr:            연환산 수익률
      mdd:            최대 낙폭 (음수)
      sortino:        소르티노 비율
      calmar:         칼마 비율 (ARR / |MDD|)
      max/min_return: 최고/최저 수익률
      by_sector:      섹터별 평균 수익률 dict
    """
    if result_df.empty:
        return {}

    buy_df = result_df[result_df["signal"] == "BUY"].copy()
    if buy_df.empty:
        return {}

    buy_df = buy_df.sort_values("date")  # 날짜 순 정렬 (누적 수익률 계산용)
    returns = buy_df["actual_return"]
    periods_per_year = 252 / hold_days  # 연간 포트폴리오 회전 횟수

    cum = (1 + returns).cumprod()  # 복리 누적 수익률
    mean_r = float(returns.mean())
    std_r = float(returns.std()) if len(returns) > 1 else 0.0

    arr = _compute_arr(cum, periods_per_year)
    mdd = _compute_mdd(cum)
    # 샤프 비율: 평균수익 / 표준편차 × √periods_per_year (무위험이자율 0 가정)
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
    """백테스트 결과 시각화 (2×3 레이아웃, PNG 저장)

    6개 서브플롯:
      [0,0] 수익률 분포 히스토그램
      [0,1] 누적 수익률 곡선 (ARR 표시)
      [0,2] 낙폭(Drawdown) 차트 (MDD 표시)
      [1,0] 섹터별 평균 수익률 막대 차트
      [1,1] 신뢰도 vs 실제 수익률 산점도
      [1,2] ARR·MDD·샤프·소르티노·칼마 핵심 지표 막대 차트
    """
    if result_df.empty:
        return

    buy_df = result_df[result_df["signal"] == "BUY"].copy()
    buy_df = buy_df.sort_values("date")
    returns = buy_df["actual_return"]
    periods_per_year = 252 / hold_days

    # 누적 수익률 및 낙폭 시계열 계산
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    drawdown = (cum - peak) / peak  # 각 시점의 고점 대비 낙폭

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

    # [0,0] 수익률 분포: 정규분포 여부와 0% 기준선 확인
    axes[0, 0].hist(returns * 100, bins=50, color="steelblue", edgecolor="white")
    axes[0, 0].axvline(0, color="red", linestyle="--")
    axes[0, 0].set_title("수익률 분포 (%)")
    axes[0, 0].set_xlabel("수익률 (%)")

    # [0,1] 누적 수익률: 전략의 복리 성장 곡선
    axes[0, 1].plot(x, cum.values, color="green", label="누적 수익")
    axes[0, 1].axhline(1, color="red", linestyle="--")
    axes[0, 1].set_title(f"누적 수익률 (ARR {arr:+.1%})")
    axes[0, 1].set_ylabel("누적 배수")
    axes[0, 1].legend()

    # [0,2] 낙폭 언더워터 차트: 고점 아래 얼마나 오래 머물렀는지
    axes[0, 2].fill_between(x, drawdown.values * 100, 0, color="tomato", alpha=0.6)
    axes[0, 2].plot(x, drawdown.values * 100, color="darkred", linewidth=0.8)
    axes[0, 2].set_title(f"낙폭 (MDD {mdd:.1%})")
    axes[0, 2].set_ylabel("낙폭 (%)")

    # [1,0] 섹터별 수익률: 어느 섹터에서 수익이 났는지
    sector_ret = buy_df.groupby("sector")["actual_return"].mean().sort_values()
    colors = ["tomato" if v < 0 else "steelblue" for v in sector_ret.values]
    axes[1, 0].barh(sector_ret.index, sector_ret.values * 100, color=colors)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_title("섹터별 평균 수익률 (%)")

    # [1,1] 신뢰도 vs 수익률 산점도: 높은 신뢰도 = 높은 수익률인지 검증
    axes[1, 1].scatter(buy_df["confidence"], returns * 100,
                       alpha=0.4, color="purple", s=20)
    axes[1, 1].axhline(0, color="red", linestyle="--")
    axes[1, 1].set_title("신뢰도 vs 실제 수익률")
    axes[1, 1].set_xlabel("신뢰도")
    axes[1, 1].set_ylabel("실제 수익률 (%)")

    # [1,2] 핵심 리스크 지표 요약 막대 차트
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
