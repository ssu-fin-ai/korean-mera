"""2025년 월별 백테스트: 전문가별 포트폴리오 선정 및 성과 평가

실행 예시:
    py backtest_2025.py                      # 전체 12개월
    py backtest_2025.py --no-precache        # OHLCV 캐시 단계 건너뜀
    py backtest_2025.py --month 1            # 특정 월만 (1~12)
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

BACKTEST_DIR = ROOT / "reports" / "backtest_2025"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = ROOT / SETTINGS["paths"]["data_cache"]

EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]
EXPERT_KOR = {
    "growth": "성장주", "value": "가치주", "theme": "테마주",
    "dividend": "배당주", "crisis": "위기종목",
}

# ── 2025년 월별 스케줄 (각 월 첫 거래일 → 월말 거래일) ───────────────────────
MONTHLY_SCHEDULE = [
    ("20250102", "20250131"),   # 1월
    ("20250203", "20250228"),   # 2월  (설날 연휴: 1/28~30)
    ("20250303", "20250331"),   # 3월  (삼일절 3/1 → 3/3 월요일)
    ("20250401", "20250430"),   # 4월
    ("20250502", "20250530"),   # 5월  (근로자의날 5/1, 어린이날 5/5)
    ("20250602", "20250630"),   # 6월  (현충일 6/6)
    ("20250701", "20250731"),   # 7월
    ("20250801", "20250829"),   # 8월  (광복절 8/15)
    ("20250901", "20250930"),   # 9월
    ("20251001", "20251031"),   # 10월 (개천절 10/3, 추석 10/5~8, 한글날 10/9)
    ("20251103", "20251128"),   # 11월 (11/1 토 → 11/3 월요일)
    ("20251201", "20251231"),   # 12월
]


# ── OHLCV 사전 캐시 ──────────────────────────────────────────────────────────

def pre_cache_ohlcv(schedule: list[tuple], collector: "KoreanStockCollector") -> None:
    tickers = collector.get_universe()
    threshold = int(len(tickers) * 0.9)

    for port_date, _eval_date in schedule:
        start = (datetime.strptime(port_date, "%Y%m%d") - timedelta(days=380)).strftime("%Y%m%d")
        cached = list(CACHE_DIR.glob(f"*_{start}_{port_date}.parquet"))

        if len(cached) >= threshold:
            logger.info(f"[{port_date}] 캐시 충분 ({len(cached)}/{len(tickers)}), 스킵")
        else:
            logger.info(f"[{port_date}] OHLCV 사전 캐시 시작 (보유={len(cached)}/{len(tickers)})")
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
    date_dir = BACKTEST_DIR / date_str
    combined_file = date_dir / "combined_portfolio.json"

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

    for expert in EXPERTS:
        picks = final_state.get(f"{expert}_picks", [])
        (date_dir / f"{expert}_picks.json").write_text(
            json.dumps(picks, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    portfolio = final_state.get("final_portfolio", [])
    combined_file.write_text(
        json.dumps(portfolio, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = final_state.get("report", "")
    (date_dir / "report.txt").write_text(report, encoding="utf-8")

    logger.info(f"[{date_str}] 완료: 통합 {len(portfolio)}종목")
    return final_state


# ── 단일 날짜 평가 ────────────────────────────────────────────────────────────

def evaluate_portfolio(date_str: str, eval_date_str: str, collector) -> dict:
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

def build_summary_report(schedule: list[tuple]) -> str:
    sep = "=" * 65
    lines = [
        sep,
        "  2025년 월별 백테스트 종합 리포트",
        f"  생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sep,
        "",
        "## 전문가별 평균 성과",
        f"{'날짜':<12} {'전문가':<10} {'종목수':>5} {'평균수익':>9} {'적중률':>7}",
        "-" * 50,
    ]

    all_rets: dict[str, list[float]] = {e: [] for e in EXPERTS + ["combined"]}
    all_hits: dict[str, list[float]] = {e: [] for e in EXPERTS + ["combined"]}

    for port_date, eval_date in schedule:
        eval_file = BACKTEST_DIR / port_date / f"eval_{eval_date}.json"
        if not eval_file.exists():
            continue
        results = json.loads(eval_file.read_text(encoding="utf-8"))
        for expert in EXPERTS + ["combined"]:
            if expert not in results:
                continue
            r = results[expert]
            label = EXPERT_KOR.get(expert, "통합")
            lines.append(
                f"{r['portfolio_date']:<12} {label:<10} {r['stock_count']:>5} "
                f"{r['avg_return']:>+8.1%} {r['hit_rate']:>6.0%}"
            )
            all_rets[expert].append(r["avg_return"])
            all_hits[expert].append(r["hit_rate"])

    lines += ["", "## 전문가별 전체 평균", "-" * 40]
    for expert in EXPERTS + ["combined"]:
        rets = all_rets[expert]
        if not rets:
            continue
        label = EXPERT_KOR.get(expert, "통합")
        lines.append(
            f"  {label:<10} 평균수익:{sum(rets)/len(rets):+.1%}  "
            f"적중률:{sum(all_hits[expert])/len(all_hits[expert]):.0%}  ({len(rets)}회)"
        )

    return "\n".join(lines)


# ── 메인 실행 ─────────────────────────────────────────────────────────────────

def run(schedule: list[tuple], precache: bool = True):
    collector = KoreanStockCollector()

    if precache:
        pre_cache_ohlcv(schedule, collector)

    logger.info(f"=== 2025년 월별 백테스트 시작 ({len(schedule)}회) ===")
    for port_date, eval_date in schedule:
        try:
            run_portfolio(port_date)
            evaluate_portfolio(port_date, eval_date, collector)
        except Exception as e:
            logger.error(f"[{port_date}] 실패: {e}")
        time.sleep(30)

    report = build_summary_report(schedule)
    report_path = BACKTEST_DIR / "summary_monthly.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"리포트 저장: {report_path}")
    sys.stdout.buffer.write((report + "\n").encode("utf-8"))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2025년 월별 백테스트")
    parser.add_argument("--no-precache", action="store_true", help="OHLCV 사전 캐시 건너뜀")
    parser.add_argument("--month", type=int, choices=range(1, 13), help="특정 월만 실행 (1~12)")
    args = parser.parse_args()

    sched = MONTHLY_SCHEDULE
    if args.month:
        sched = [e for e in MONTHLY_SCHEDULE if int(e[0][4:6]) == args.month]
        logger.info(f"{args.month}월만 실행: {sched}")

    run(sched, precache=not args.no_precache)
