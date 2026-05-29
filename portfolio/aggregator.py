"""전문가 에이전트 신호 집계 → 종목 랭킹 & 포트폴리오 생성"""

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from config import SETTINGS

SIGNAL_SCORE = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}


# ── 기존 단일 종목 집계 (build_history_db / update_today 호환용) ───────────

@dataclass
class StockSignal:
    ticker: str
    name: str
    sector: str
    date: str
    gate_result: dict = field(default_factory=dict)
    expert_results: list[dict] = field(default_factory=list)
    final_signal: str = "HOLD"
    final_score: float = 0.0
    final_confidence: float = 0.5
    avg_target_return: float = 0.0
    pattern_type: str = ""


def aggregate(stock_signal: StockSignal) -> StockSignal:
    """전문가 결과를 가중 평균으로 집계 (단일 종목용)"""
    results = stock_signal.expert_results
    if not results:
        return stock_signal

    gate_conf = stock_signal.gate_result.get("confidence", 0.5)
    scores, confidences, target_returns = [], [], []
    for r in results:
        sig = r.get("signal", "HOLD")
        conf = float(r.get("confidence", 0.5))
        target = float(r.get("target_return", 0.0))
        scores.append(SIGNAL_SCORE.get(sig, 0.0) * conf)
        confidences.append(conf)
        target_returns.append(target)

    weighted_score = sum(scores) / len(scores) * gate_conf
    stock_signal.final_signal = (
        "BUY" if weighted_score > 0.3 else
        "SELL" if weighted_score < -0.3 else "HOLD"
    )
    stock_signal.final_score = weighted_score
    stock_signal.final_confidence = sum(confidences) / len(confidences)
    stock_signal.avg_target_return = sum(target_returns) / len(target_returns)
    stock_signal.pattern_type = stock_signal.gate_result.get("pattern_type", "")
    return stock_signal


def build_portfolio(
    signals: list[StockSignal],
    top_n: int = 20,
    min_confidence: float = 0.60,
) -> pd.DataFrame:
    """BUY 신호 종목 중 상위 N개 포트폴리오 구성"""
    rows = [
        {
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
        }
        for s in signals
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    buy_df = (
        df[(df["signal"] == "BUY") & (df["confidence"] >= min_confidence)]
        .sort_values("score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    buy_df.index += 1
    return buy_df


def build_report(portfolio: pd.DataFrame, date: str) -> str:
    """포트폴리오 텍스트 리포트 생성 (단일 종목 파이프라인용)"""
    if portfolio.empty:
        return f"[{date}] 조건 충족 BUY 종목 없음"
    lines = [f"=== 한국주식 MERA 포트폴리오 [{date}] ===", f"총 {len(portfolio)}개 종목\n"]
    for rank, row in portfolio.iterrows():
        lines.append(f"#{rank:02d} {row['ticker']} {row['name']} [{row['sector']}]")
        lines.append(
            f"     신호:{row['signal']} 점수:{row['score']:.2f} "
            f"신뢰:{row['confidence']:.0%} 목표:{row['target_return']:+.1%}"
        )
        lines.append(f"     패턴: {row['pattern_type']}")
        for expert, reason in zip(row["experts"], row["reasons"]):
            lines.append(f"     [{expert}] {reason}")
        lines.append("")
    return "\n".join(lines)


# ── LangGraph 집계 노드 ───────────────────────────────────────────────────────

def aggregator_node(state: dict) -> dict:
    """5개 전문가 picks 통합 → 종목별 복합 스코어링 + 포트폴리오 구성"""
    all_picks: list[dict] = []
    for expert in ["growth", "value", "theme", "dividend", "crisis"]:
        all_picks.extend(state.get(f"{expert}_picks", []))

    ticker_picks: dict[str, list[dict]] = defaultdict(list)
    for pick in all_picks:
        ticker = pick.get("ticker", "")
        if ticker:
            ticker_picks[ticker].append(pick)

    cfg = SETTINGS["portfolio"]
    min_conf = float(cfg.get("signal_threshold", 0.60))
    top_n = int(cfg.get("top_n", 20))

    final_portfolio: list[dict] = []

    gate_weights_map: dict = state.get("gate_weights_map", {})

    for ticker, picks in ticker_picks.items():
        buy_picks = [
            p for p in picks
            if p.get("signal") == "BUY" and float(p.get("confidence", 0)) >= min_conf
        ]
        if not buy_picks:
            continue

        experts = list({p["expert"] for p in buy_picks})
        n = len(buy_picks)
        avg_conf = round(sum(float(p.get("confidence", 0)) for p in buy_picks) / n, 3)
        avg_score = round(sum(int(p.get("score", 5)) for p in buy_picks) / n, 1)
        avg_ret = round(sum(float(p.get("target_return", 0)) for p in buy_picks) / n, 4)
        avg_days = round(sum(int(p.get("horizon_days", 10)) for p in buy_picks) / n)

        # GateNet 가중치 반영: 전문가별 confidence×score×gate_weight 가중합
        gate_w = gate_weights_map.get(ticker, {})
        weighted_scores = [
            float(p.get("confidence", 0.5)) * int(p.get("score", 5))
            * gate_w.get(p["expert"], 1.0)
            for p in buy_picks
        ]
        # 복합 점수: gate 가중 평균 + 전문가 동의 수 보너스 (전문가당 +0.5)
        composite = round(sum(weighted_scores) / n + len(experts) * 0.5, 2)

        # 대표 근거: 신뢰도×매력도 최고 전문가
        best = max(buy_picks, key=lambda p: float(p.get("confidence", 0)) * int(p.get("score", 5)))

        # pros / cons 통합 (중복 제거, 순서 유지)
        seen_pros: set[str] = set()
        seen_cons: set[str] = set()
        unique_pros: list[str] = []
        unique_cons: list[str] = []
        for p in buy_picks:
            for x in p.get("pros", []):
                if x and x not in seen_pros:
                    seen_pros.add(x)
                    unique_pros.append(x)
            for x in p.get("cons", []):
                if x and x not in seen_cons:
                    seen_cons.add(x)
                    unique_cons.append(x)

        final_portfolio.append({
            "ticker": ticker,
            "name": picks[0].get("name", ticker),
            "signal": "BUY",
            "composite_score": composite,
            "expert_count": len(experts),
            "experts": experts,
            "avg_confidence": avg_conf,
            "avg_score": avg_score,
            "avg_target_return": avg_ret,
            "avg_horizon_days": avg_days,
            "reason": best.get("reason", ""),
            "pros": unique_pros[:6],
            "cons": unique_cons[:4],
        })

    final_portfolio.sort(key=lambda x: x["composite_score"], reverse=True)
    final_portfolio = final_portfolio[:top_n]

    logger.info(
        f"집계 완료: {len(final_portfolio)}개 BUY 종목 "
        f"(전체 분석 {len(ticker_picks)}개 중)"
    )
    return {"final_portfolio": final_portfolio}


def reporter_node(state: dict) -> dict:
    """final_portfolio → 텍스트 리포트 생성"""
    portfolio = state.get("final_portfolio", [])
    date = state.get("date", "")
    today_dash = f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date

    if not portfolio:
        return {"report": f"[{today_dash}] 조건 충족 BUY 종목 없음"}

    sep = "=" * 58
    lines = [
        sep,
        f"  한국주식 MERA 포트폴리오  [{today_dash}]",
        f"  BUY 종목: {len(portfolio)}개",
        sep,
        "",
    ]

    for rank, item in enumerate(portfolio, 1):
        experts_str = " · ".join(item["experts"])
        lines += [
            f"#{rank:02d}  {item['ticker']}  {item['name']}",
            f"    복합점수:{item['composite_score']:.1f}  "
            f"신뢰:{item['avg_confidence']:.0%}  "
            f"목표:{item['avg_target_return']:+.1%}/{item['avg_horizon_days']}일",
            f"    전문가: [{experts_str}]",
            f"    근거: {item['reason']}",
            "    [긍정 요인]",
        ]
        for pro in item["pros"]:
            lines.append(f"      + {pro}")
        lines.append("    [리스크]")
        for con in item["cons"]:
            lines.append(f"      - {con}")
        lines.append("")

    return {"report": "\n".join(lines)}


# ── 포트폴리오 DataFrame 변환 (PortfolioStore 저장용) ─────────────────────────

def portfolio_to_df(portfolio: list[dict], date: str) -> pd.DataFrame:
    """LangGraph final_portfolio → PortfolioStore 호환 DataFrame"""
    if not portfolio:
        return pd.DataFrame()
    rows = [
        {
            "ticker": item["ticker"],
            "name": item["name"],
            "sector": "",
            "date": date,
            "signal": item["signal"],
            "score": item["composite_score"],
            "confidence": item["avg_confidence"],
            "target_return": item["avg_target_return"],
        }
        for item in portfolio
    ]
    df = pd.DataFrame(rows)
    df.index = range(1, len(df) + 1)
    return df
