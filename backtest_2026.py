"""2026년 백테스트: 전문가별 월별/주별 포트폴리오 선정 및 성과 평가

실행 예시:
    py backtest_2026.py --freq monthly          # 월별 5회
    py backtest_2026.py --freq weekly           # 주별 ~20회
    py backtest_2026.py --freq both             # 월별 + 주별 (기본값)
    py backtest_2026.py --freq monthly --no-precache  # OHLCV 캐시 단계 건너뜀
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from loguru import logger

from config import ROOT, SETTINGS
from data.collector import KoreanStockCollector
from portfolio.evaluator import compute_actual_return

BACKTEST_DIR = ROOT / "reports" / "backtest_2026"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT / SETTINGS["paths"]["data_cache"]

EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]
EXPERT_KOR = {
    "growth": "성장주", "value": "가치주", "theme": "테마주",
    "dividend": "배당주", "crisis": "위기종목",
}

# ── 백테스트 날짜 정의 ────────────────────────────────────────────────────────

# 월별: 각 월 첫 번째 거래일 → 평가일(월말)
MONTHLY_SCHEDULE = [
    ("20260102", "20260130"),  # 1월
    ("20260202", "20260227"),  # 2월
    ("20260302", "20260331"),  # 3월
    ("20260401", "20260430"),  # 4월
    ("20260504", "20260523"),  # 5월 (당일 기준)
]

# 주별: 매주 월요일 → 5영업일 후
def _gen_weekly_schedule():
    schedule = []
    mondays = pd.date_range("2026-01-05", "2026-05-18", freq="W-MON")
    for mon in mondays:
        port_date = mon.strftime("%Y%m%d")
        eval_dt = mon + pd.tseries.offsets.BDay(5)
        # 평가일이 오늘(2026-05-23) 이후면 오늘로 제한
        if eval_dt > pd.Timestamp("2026-05-23"):
            eval_dt = pd.Timestamp("2026-05-23")
        eval_date = eval_dt.strftime("%Y%m%d")
        schedule.append((port_date, eval_date))
    return schedule

WEEKLY_SCHEDULE = _gen_weekly_schedule()


# ── OHLCV 사전 캐시 ──────────────────────────────────────────────────────────

def pre_cache_ohlcv(schedule: list[tuple], collector: "KoreanStockCollector") -> None:
    """스크리너가 필요한 OHLCV 파일을 미리 다운로드하고, 빈 포트폴리오 스텁을 삭제한다."""
    tickers = collector.get_universe()
    threshold = int(len(tickers) * 0.9)

    for port_date, _eval_date in schedule:
        start = (datetime.strptime(port_date, "%Y%m%d") - timedelta(days=380)).strftime("%Y%m%d")
        cached = list(CACHE_DIR.glob(f"*_{start}_{port_date}.parquet"))

        if len(cached) >= threshold:
            logger.info(f"[{port_date}] 캐시 충분 ({len(cached)}/{len(tickers)}), 스킵")
        else:
            logger.info(f"[{port_date}] OHLCV 사전 캐시 시작 (start={start}, 보유={len(cached)}/{len(tickers)})")
            for i, ticker in enumerate(tickers):
                cache_path = CACHE_DIR / f"{ticker}_{start}_{port_date}.parquet"
                if cache_path.exists():
                    continue
                try:
                    collector.get_ohlcv(ticker, start, port_date)
                except Exception as e:
                    logger.debug(f"  {ticker} 캐시 실패: {e}")
                if (i + 1) % 50 == 0:
                    logger.info(f"  [{port_date}] {i+1}/{len(tickers)} 완료")
                time.sleep(0.3)
            logger.info(f"[{port_date}] OHLCV 사전 캐시 완료")

        # 빈 포트폴리오 스텁 삭제 → run_portfolio() 재실행 허용
        combined = BACKTEST_DIR / port_date / "combined_portfolio.json"
        if combined.exists():
            try:
                data = json.loads(combined.read_text(encoding="utf-8"))
                if len(data) == 0:
                    logger.info(f"[{port_date}] 빈 포트폴리오 삭제 → 재실행 예정")
                    shutil.rmtree(BACKTEST_DIR / port_date)
            except Exception:
                pass


# ── 단일 날짜 포트폴리오 실행 ─────────────────────────────────────────────────

def run_portfolio(date_str: str) -> dict:
    """LangGraph 파이프라인 실행 → 전문가별 picks + 통합 포트폴리오 반환"""
    date_dir = BACKTEST_DIR / date_str
    combined_file = date_dir / "combined_portfolio.json"

    # 이미 실행된 날짜는 스킵
    if combined_file.exists():
        logger.info(f"[{date_str}] 캐시 사용 (이미 실행됨)")
        state = {}
        for expert in EXPERTS:
            f = date_dir / f"{expert}_picks.json"
            if f.exists():
                state[f"{expert}_picks"] = json.loads(f.read_text(encoding="utf-8"))
        state["final_portfolio"] = json.loads(combined_file.read_text(encoding="utf-8"))
        state["date"] = date_str
        return state

    logger.info(f"[{date_str}] 포트폴리오 선정 시작")
    date_dir.mkdir(exist_ok=True)

    from agents.graph import build_mera_graph
    graph = build_mera_graph()
    final_state = graph.invoke({"date": date_str})

    # 전문가별 picks 저장
    for expert in EXPERTS:
        picks = final_state.get(f"{expert}_picks", [])
        (date_dir / f"{expert}_picks.json").write_text(
            json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 통합 포트폴리오 저장
    portfolio = final_state.get("final_portfolio", [])
    combined_file.write_text(
        json.dumps(portfolio, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # 리포트 저장
    report = final_state.get("report", "")
    (date_dir / "report.txt").write_text(report, encoding="utf-8")

    logger.info(f"[{date_str}] 완료: 통합 {len(portfolio)}종목")
    return final_state


# ── 단일 날짜 평가 ────────────────────────────────────────────────────────────

def evaluate_portfolio(date_str: str, eval_date_str: str, collector) -> dict:
    """전문가별 + 통합 포트폴리오 실제 수익률 계산"""
    date_dir = BACKTEST_DIR / date_str
    eval_file = date_dir / f"eval_{eval_date_str}.json"

    if eval_file.exists():
        logger.info(f"[{date_str}→{eval_date_str}] 평가 캐시 사용")
        return json.loads(eval_file.read_text(encoding="utf-8"))

    port_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    eval_date = f"{eval_date_str[:4]}-{eval_date_str[4:6]}-{eval_date_str[6:]}"

    results: dict = {}
    targets = [(e, f"{e}_picks.json") for e in EXPERTS]
    targets.append(("combined", "combined_portfolio.json"))

    for expert, filename in targets:
        portfolio_file = date_dir / filename
        if not portfolio_file.exists():
            continue
        picks = json.loads(portfolio_file.read_text(encoding="utf-8"))
        buy_picks = [p for p in picks if p.get("signal") == "BUY"]
        if not buy_picks:
            continue

        stock_results = []
        for pick in buy_picks:
            ret = compute_actual_return(collector, pick["ticker"], port_date, eval_date)
            if ret is None:
                continue
            target_ret = float(pick.get("target_return") or pick.get("avg_target_return") or 0)
            stock_results.append({
                "ticker": pick["ticker"],
                "name": pick.get("name", ""),
                "reason": pick.get("reason", ""),
                "pros": pick.get("pros", []),
                "cons": pick.get("cons", []),
                "target_return": target_ret,
                "actual_return": ret,
                "hit": ret > 0,
            })

        if not stock_results:
            continue

        returns = [s["actual_return"] for s in stock_results]
        avg_ret = sum(returns) / len(returns)
        hit_rate = sum(1 for s in stock_results if s["hit"]) / len(stock_results)

        results[expert] = {
            "portfolio_date": port_date,
            "eval_date": eval_date,
            "stocks": stock_results,
            "avg_return": avg_ret,
            "hit_rate": hit_rate,
            "stock_count": len(stock_results),
        }

    if results:
        eval_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(
            f"[{date_str}→{eval_date_str}] 평가 완료: "
            + ", ".join(
                f"{EXPERT_KOR.get(e, e)} {v['avg_return']:+.1%}/{v['hit_rate']:.0%}"
                for e, v in results.items() if e != "combined"
            )
        )

    return results


# ── 종합 리포트 생성 ──────────────────────────────────────────────────────────

def build_summary_report(schedule: list[tuple], freq_label: str) -> str:
    """백테스트 종합 리포트 텍스트 생성"""
    sep = "=" * 65

    lines = [
        sep,
        f"  2026년 {freq_label} 백테스트 종합 리포트",
        f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sep,
        "",
    ]

    # ── 성과 요약 테이블 ──
    lines += [
        "## 전문가별 평균 성과",
        f"{'날짜':<12} {'전문가':<10} {'종목수':>5} {'평균수익':>9} {'적중률':>7}",
        "-" * 50,
    ]

    all_expert_returns: dict[str, list[float]] = {e: [] for e in EXPERTS + ["combined"]}
    all_expert_hits: dict[str, list[float]] = {e: [] for e in EXPERTS + ["combined"]}

    for port_date, eval_date in schedule:
        eval_file = BACKTEST_DIR / port_date / f"eval_{eval_date}.json"
        if not eval_file.exists():
            continue
        results = json.loads(eval_file.read_text(encoding="utf-8"))

        for expert in EXPERTS + ["combined"]:
            if expert not in results:
                continue
            r = results[expert]
            label = EXPERT_KOR.get(expert, expert)
            lines.append(
                f"{r['portfolio_date']:<12} {label:<10} {r['stock_count']:>5} "
                f"{r['avg_return']:>+8.1%} {r['hit_rate']:>6.0%}"
            )
            all_expert_returns[expert].append(r["avg_return"])
            all_expert_hits[expert].append(r["hit_rate"])

    lines += ["", "## 전문가별 전체 평균", "-" * 40]
    for expert in EXPERTS + ["combined"]:
        rets = all_expert_returns[expert]
        hits = all_expert_hits[expert]
        if not rets:
            continue
        label = EXPERT_KOR.get(expert, "통합")
        avg_r = sum(rets) / len(rets)
        avg_h = sum(hits) / len(hits)
        lines.append(f"  {label:<10} 평균수익:{avg_r:+.1%}  적중률:{avg_h:.0%}  ({len(rets)}회)")

    # ── 날짜별 상세 ──
    for port_date, eval_date in schedule:
        eval_file = BACKTEST_DIR / port_date / f"eval_{eval_date}.json"
        if not eval_file.exists():
            continue
        results = json.loads(eval_file.read_text(encoding="utf-8"))
        port_dash = f"{port_date[:4]}-{port_date[4:6]}-{port_date[6:]}"
        eval_dash = f"{eval_date[:4]}-{eval_date[4:6]}-{eval_date[6:]}"

        lines += [
            "",
            sep,
            f"  포트폴리오: {port_dash} → 평가: {eval_dash}",
            sep,
        ]

        for expert in EXPERTS + ["combined"]:
            if expert not in results:
                continue
            r = results[expert]
            label = EXPERT_KOR.get(expert, "통합")
            lines += [
                "",
                f"▶ {label} 전문가  "
                f"평균수익:{r['avg_return']:+.1%}  적중률:{r['hit_rate']:.0%}  "
                f"({r['stock_count']}종목)",
            ]

            stocks = sorted(r["stocks"], key=lambda x: x["actual_return"], reverse=True)
            for i, s in enumerate(stocks, 1):
                hit_mark = "O" if s["hit"] else "X"
                lines += [
                    f"  {i:02d}. {hit_mark} {s['ticker']} {s['name']}  "
                    f"목표:{s['target_return']:+.1%}  실제:{s['actual_return']:+.1%}",
                    f"      근거: {s['reason']}",
                ]
                if s.get("pros"):
                    lines.append("      [긍정]")
                    for pro in s["pros"][:3]:
                        lines.append(f"        + {pro}")
                if s.get("cons"):
                    lines.append("      [리스크]")
                    for con in s["cons"][:2]:
                        lines.append(f"        - {con}")

    return "\n".join(lines)


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def run(freq: str = "both", precache: bool = True):
    collector = KoreanStockCollector()

    if freq in ("monthly", "both"):
        if precache:
            pre_cache_ohlcv(MONTHLY_SCHEDULE, collector)
        logger.info(f"=== 월별 백테스트 시작 ({len(MONTHLY_SCHEDULE)}회) ===")
        for port_date, eval_date in MONTHLY_SCHEDULE:
            try:
                run_portfolio(port_date)
                evaluate_portfolio(port_date, eval_date, collector)
            except Exception as e:
                logger.error(f"[{port_date}] 실패: {e}")
            time.sleep(30)  # KRX 세션 유지용 딜레이

        report = build_summary_report(MONTHLY_SCHEDULE, "월별")
        report_path = BACKTEST_DIR / "summary_monthly.txt"
        report_path.write_text(report, encoding="utf-8")
        logger.info(f"월별 리포트 저장: {report_path}")
        sys.stdout.buffer.write((report + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()

    if freq in ("weekly", "both"):
        if precache:
            pre_cache_ohlcv(WEEKLY_SCHEDULE, collector)
        logger.info(f"=== 주별 백테스트 시작 ({len(WEEKLY_SCHEDULE)}회) ===")
        for port_date, eval_date in WEEKLY_SCHEDULE:
            try:
                run_portfolio(port_date)
                evaluate_portfolio(port_date, eval_date, collector)
            except Exception as e:
                logger.error(f"[{port_date}] 실패: {e}")
            time.sleep(20)  # KRX 세션 유지용 딜레이

        report = build_summary_report(WEEKLY_SCHEDULE, "주별")
        report_path = BACKTEST_DIR / "summary_weekly.txt"
        report_path.write_text(report, encoding="utf-8")
        logger.info(f"주별 리포트 저장: {report_path}")
        sys.stdout.buffer.write((report + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026년 백테스트")
    parser.add_argument(
        "--freq",
        choices=["monthly", "weekly", "both"],
        default="both",
        help="실행 주기 (기본값: both)",
    )
    parser.add_argument(
        "--no-precache",
        action="store_true",
        help="OHLCV 사전 캐시 단계 건너뜀",
    )
    args = parser.parse_args()
    run(args.freq, precache=not args.no_precache)
