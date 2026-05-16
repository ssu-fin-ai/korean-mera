"""전문가 에이전트 신호 집계 → 종목 랭킹 & 포트폴리오 생성"""

from dataclasses import dataclass, field

import pandas as pd

SIGNAL_SCORE = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}


@dataclass
class StockSignal:
    ticker: str
    name: str
    sector: str
    date: str
    gate_result: dict = field(default_factory=dict)
    expert_results: list[dict] = field(default_factory=list)

    # 집계 결과
    final_signal: str = "HOLD"
    final_score: float = 0.0
    final_confidence: float = 0.5
    avg_target_return: float = 0.0
    pattern_type: str = ""


def aggregate(stock_signal: StockSignal) -> StockSignal:
    """전문가 결과를 가중 평균으로 집계"""
    results = stock_signal.expert_results
    if not results:
        return stock_signal

    gate_conf = stock_signal.gate_result.get("confidence", 0.5)

    # 각 전문가 신호를 수치화
    scores, confidences, target_returns = [], [], []
    for r in results:
        sig = r.get("signal", "HOLD")
        conf = float(r.get("confidence", 0.5))
        target = float(r.get("target_return", 0.0))

        scores.append(SIGNAL_SCORE.get(sig, 0.0) * conf)
        confidences.append(conf)
        target_returns.append(target)

    # GateNet 신뢰도를 가중치로 활용
    weighted_score = sum(scores) / len(scores) * gate_conf
    avg_conf = sum(confidences) / len(confidences)
    avg_target = sum(target_returns) / len(target_returns)

    # 최종 시그널 결정
    if weighted_score > 0.3:
        final_signal = "BUY"
    elif weighted_score < -0.3:
        final_signal = "SELL"
    else:
        final_signal = "HOLD"

    stock_signal.final_signal = final_signal
    stock_signal.final_score = weighted_score
    stock_signal.final_confidence = avg_conf
    stock_signal.avg_target_return = avg_target
    stock_signal.pattern_type = stock_signal.gate_result.get("pattern_type", "")
    return stock_signal


def build_portfolio(
    signals: list[StockSignal],
    top_n: int = 20,
    min_confidence: float = 0.60,
) -> pd.DataFrame:
    """BUY 신호 종목 중 상위 N개 포트폴리오 구성"""
    rows = []
    for s in signals:
        rows.append({
            "ticker": s.ticker,
            "name": s.name,
            "sector": s.sector,
            "date": s.date,
            "signal": s.final_signal,
            "score": s.final_score,
            "confidence": s.final_confidence,
            "target_return": s.avg_target_return,
            "pattern_type": s.pattern_type,
            "experts": [r.get("expert", "") for r in s.expert_results],
            "reasons": [r.get("reason", "") for r in s.expert_results],
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    buy_df = df[
        (df["signal"] == "BUY") & (df["confidence"] >= min_confidence)
    ].sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    buy_df.index += 1  # 1-based 랭킹
    return buy_df


def build_report(portfolio: pd.DataFrame, date: str) -> str:
    """포트폴리오 텍스트 리포트 생성"""
    if portfolio.empty:
        return f"[{date}] 조건 충족 BUY 종목 없음"

    lines = [f"=== 한국주식 MERA 포트폴리오 [{date}] ===",
             f"총 {len(portfolio)}개 종목\n"]

    for rank, row in portfolio.iterrows():
        lines.append(
            f"#{rank:02d} {row['ticker']} {row['name']} [{row['sector']}]"
        )
        lines.append(
            f"     신호:{row['signal']} 점수:{row['score']:.2f} "
            f"신뢰:{row['confidence']:.0%} 목표:{row['target_return']:+.1%}"
        )
        lines.append(f"     패턴: {row['pattern_type']}")
        for expert, reason in zip(row["experts"], row["reasons"]):
            lines.append(f"     [{expert}] {reason}")
        lines.append("")

    return "\n".join(lines)
