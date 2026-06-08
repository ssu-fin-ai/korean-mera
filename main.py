"""MERA 한국주식 시스템 진입점

커맨드라인 인터페이스(CLI)로 모든 실행 모드를 제어한다.

사용법:
  python main.py --mode build_db              # 최초 히스토리 DB 구축
  python main.py --mode run                   # 오늘 신호 생성 (월간 기준)
  python main.py --mode run --date 20240115   # 특정 날짜 신호 생성
  python main.py --mode run --horizon weekly  # 주간 분석 (5일 RAG)
  python main.py --mode schedule              # 평일 자동 스케줄러 실행
  python main.py --mode backtest              # 과거 포트폴리오 백테스트
  python main.py --mode evaluate --date ...   # 특정 날짜 포트폴리오 평가
"""

import argparse
from loguru import logger
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (상대 임포트 지원)
sys.path.insert(0, str(Path(__file__).parent))

from config import ROOT

# 환경변수 검증 대상 (없으면 실행 불가)
REQUIRED_ENV_VARS = ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "KRX_ID", "KRX_PW"]


def check_config():
    """.env 파일 존재 여부 + 필수 환경변수 설정 여부 검증

    .env 파일이 없거나 필수 키가 없으면 안내 메시지를 출력하고 종료.
    """
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
            "  KRX_ID=your_krx_id\n"
            "  KRX_PW=your_krx_password\n"
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
    """Phase 1: ChromaDB 히스토리 패턴 DB 구축 (최초 1회)

    --years N 지정 시 N년치, 미지정 시 settings.yaml history_years 사용.
    """
    from scheduler.pipeline import MERAPipeline
    years = args.years
    logger.info(f"히스토리 DB 구축 시작 ({years}년)")
    pipeline = MERAPipeline()
    pipeline.build_history_db(years=years)


def cmd_run(args):
    """Phase 2: LangGraph MERA 그래프 실행 → 포트폴리오 리포트 출력

    --date YYYYMMDD: 특정 날짜 분석 (기본값: 오늘)
    --horizon weekly|monthly: 5일/20일 RAG 기준 (기본값: monthly)
    """
    from scheduler.pipeline import MERAPipeline
    date = args.date if hasattr(args, "date") and args.date else None
    horizon = args.horizon if hasattr(args, "horizon") and args.horizon else "monthly"
    pipeline = MERAPipeline()
    report = pipeline.run_daily(date=date, horizon=horizon)
    print(report)


def cmd_schedule(_args):
    """평일 자동 스케줄러 시작 (settings.yaml data.update_time에 설정된 시각)"""
    from scheduler.daily_runner import start_scheduler
    start_scheduler()


def cmd_evaluate(args):
    """특정 날짜 포트폴리오 수익률 평가 (5영업일 전 포트폴리오 기준)

    Supabase에 저장된 포트폴리오를 찾아 실제 수익률을 계산하고
    LLM 평가 결과와 함께 DB에 저장한다.
    """
    from scheduler.pipeline import MERAPipeline
    from portfolio.evaluator import run_evaluation

    pipeline = MERAPipeline()
    date = args.date or __import__("datetime").datetime.today().strftime("%Y%m%d")
    today_dash = f"{date[:4]}-{date[4:6]}-{date[6:]}"

    # 5영업일 전 포트폴리오 날짜 탐색
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
    """과거 포트폴리오 CSV 파일 기반 백테스트

    reports/ 폴더의 portfolio_*.csv를 모두 불러와
    --hold_days N일 보유 후 실제 수익률을 계산하고
    ARR·MDD·샤프·소르티노·칼마 등 성과 지표를 출력한다.
    """
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
    plot_results(result_df)  # reports/backtest_result.png 저장

    import json
    # by_sector는 dict이므로 제외하고 출력
    print(json.dumps({k: v for k, v in metrics.items() if k != "by_sector"}, indent=2))


# 실행 모드 → 핸들러 함수 매핑
COMMANDS = {
    "build_db": cmd_build_db,
    "run": cmd_run,
    "evaluate": cmd_evaluate,
    "schedule": cmd_schedule,
    "backtest": cmd_backtest,
}


def main():
    """CLI 진입점: 인자 파싱 → 설정 검증 → 로그 설정 → 명령 실행"""
    parser = argparse.ArgumentParser(description="MERA 한국주식 AI 에이전트")
    parser.add_argument("--mode", choices=list(COMMANDS.keys()),
                        default="run", help="실행 모드")
    parser.add_argument("--date", type=str, default=None,
                        help="분석 날짜 (YYYYMMDD, 기본값: 오늘)")
    parser.add_argument("--years", type=int, default=None,
                        help="히스토리 DB 구축 기간 (년, 기본값: settings.yaml history_years)")
    parser.add_argument("--horizon", choices=["weekly", "monthly"], default="monthly",
                        help="분석 기간 (weekly=5일 RAG, monthly=20일 RAG)")
    parser.add_argument("--hold_days", type=int, default=5,
                        help="백테스트 보유 기간 (일)")
    args = parser.parse_args()

    # API 키 및 .env 파일 검증
    check_config()

    # 일별 로테이션 로그 파일 (30일 보관)
    logger.add(ROOT / "logs" / "mera_{time}.log",
               rotation="1 day", retention="30 days", level="INFO")

    COMMANDS[args.mode](args)


if __name__ == "__main__":
    main()
