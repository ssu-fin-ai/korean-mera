"""MERA 한국주식 시스템 진입점

사용법:
  python main.py --mode build_db          # 최초 히스토리 DB 구축
  python main.py --mode run               # 오늘 신호 생성
  python main.py --mode run --date 20240115  # 특정 날짜 신호 생성
  python main.py --mode schedule          # 자동 스케줄러 실행
  python main.py --mode backtest          # 백테스트
"""

import argparse
from loguru import logger
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import ROOT

REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY"]


def check_config():
    import os
    from dotenv import load_dotenv

    env_file = Path(__file__).parent / ".env"
    if not env_file.exists():
        print(
            "\n[오류] .env 파일이 없습니다.\n"
            "아래 명령으로 생성 후 API 키를 입력하세요:\n\n"
            "  cp .env.example .env\n\n"
            "필수 항목:\n"
            "  ANTHROPIC_API_KEY=sk-ant-...\n"
            "  OPENAI_API_KEY=sk-...\n"
        )
        sys.exit(1)

    load_dotenv(env_file)

    missing = [k for k in REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        print(
            f"\n[오류] .env 파일에 필수 항목이 없습니다: {', '.join(missing)}\n"
            ".env 파일을 열어 아래 항목을 입력하세요:\n\n"
            + "\n".join(f"  {k}=..." for k in missing)
            + "\n"
        )
        sys.exit(1)


def cmd_build_db(args):
    from scheduler.pipeline import MERAPipeline
    years = int(args.years) if hasattr(args, "years") and args.years else 5
    logger.info(f"히스토리 DB 구축 시작 ({years}년)")
    pipeline = MERAPipeline()
    pipeline.build_history_db(years=years)


def cmd_run(args):
    from scheduler.pipeline import MERAPipeline
    date = args.date if hasattr(args, "date") and args.date else None
    pipeline = MERAPipeline()
    report = pipeline.run_daily(date=date)
    print(report)


def cmd_schedule(_args):
    from scheduler.daily_runner import start_scheduler
    start_scheduler()


def cmd_evaluate(args):
    from scheduler.pipeline import MERAPipeline
    from portfolio.evaluator import run_evaluation

    pipeline = MERAPipeline()
    date = args.date or __import__("datetime").datetime.today().strftime("%Y%m%d")
    today_dash = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    prev_date = pipeline._get_weekly_portfolio_date(today_dash)
    if not prev_date:
        logger.error("평가할 포트폴리오가 DB에 없습니다. 먼저 --mode run 실행 필요")
        return

    result = run_evaluation(prev_date, today_dash, pipeline.collector)
    if result:
        print(f"\n=== 포트폴리오 평가 결과 [{prev_date} → {today_dash}] ===")
        print(f"종목수: {result['stock_count']} | 평균수익: {result['avg_return']:+.1%} | 적중률: {result['hit_rate']:.0%}")
        print(f"\n[LLM 종합 평가]\n{result['summary']}")
    else:
        print("평가 결과 없음 (이미 평가됨 or 데이터 부족)")


def cmd_backtest(args):
    from evaluation.backtest import (
        load_portfolios, compute_returns, compute_metrics, plot_results
    )
    from data.collector import KoreanStockCollector

    portfolio_df = load_portfolios()
    if portfolio_df.empty:
        logger.error("포트폴리오 데이터 없음. 먼저 --mode run 실행 필요")
        return

    tickers = portfolio_df["ticker"].unique().tolist()
    collector = KoreanStockCollector()
    hold_days = int(args.hold_days) if hasattr(args, "hold_days") and args.hold_days else 5

    logger.info(f"{len(tickers)}개 종목 가격 데이터 로드 중...")
    price_data = {}
    for t in tickers:
        try:
            df = collector.get_ohlcv(t, "20200101", "20251231")
            if not df.empty:
                price_data[t] = df
        except Exception:
            pass

    result_df = compute_returns(portfolio_df, price_data, hold_days=hold_days)
    metrics = compute_metrics(result_df)
    plot_results(result_df)

    import json
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_sector"}, indent=2))


COMMANDS = {
    "build_db": cmd_build_db,
    "run": cmd_run,
    "evaluate": cmd_evaluate,
    "schedule": cmd_schedule,
    "backtest": cmd_backtest,
}


def main():
    parser = argparse.ArgumentParser(description="MERA 한국주식 AI 에이전트")
    parser.add_argument("--mode", choices=list(COMMANDS.keys()),
                        default="run", help="실행 모드")
    parser.add_argument("--date", type=str, default=None,
                        help="분석 날짜 (YYYYMMDD, 기본값: 오늘)")
    parser.add_argument("--years", type=int, default=5,
                        help="히스토리 DB 구축 기간 (년)")
    parser.add_argument("--hold_days", type=int, default=5,
                        help="백테스트 보유 기간 (일)")
    args = parser.parse_args()

    check_config()

    logger.add(ROOT / "logs" / "mera_{time}.log",
               rotation="1 day", retention="30 days", level="INFO")

    COMMANDS[args.mode](args)


if __name__ == "__main__":
    main()
