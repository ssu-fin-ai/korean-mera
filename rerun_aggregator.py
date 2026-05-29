"""aggregator만 재실행 (LLM 호출 없이) + eval 재실행

기존 {expert}_picks.json 을 읽어 aggregator_node 만 다시 돌리고
combined_portfolio.json 을 top_n=5 기준으로 덮어씀.
이후 evaluate_portfolio 를 재실행해 eval_{date}.json 을 갱신.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from loguru import logger

from config import ROOT
from portfolio.aggregator import aggregator_node
from data.collector import KoreanStockCollector
from backtest_2026 import (
    MONTHLY_SCHEDULE, WEEKLY_SCHEDULE, BACKTEST_DIR, evaluate_portfolio
)

EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]

ALL_SCHEDULE = list({p: e for p, e in MONTHLY_SCHEDULE + WEEKLY_SCHEDULE}.items())


def rerun_aggregator():
    updated = 0
    for date_dir in sorted(BACKTEST_DIR.glob("2026*")):
        picks_exist = all((date_dir / f"{exp}_picks.json").exists() for exp in EXPERTS)
        if not picks_exist:
            logger.warning(f"[{date_dir.name}] 전문가 picks 없음, 스킵")
            continue

        state = {"date": date_dir.name}
        for exp in EXPERTS:
            state[f"{exp}_picks"] = json.loads(
                (date_dir / f"{exp}_picks.json").read_text(encoding="utf-8")
            )

        result = aggregator_node(state)
        portfolio = result["final_portfolio"]

        (date_dir / "combined_portfolio.json").write_text(
            json.dumps(portfolio, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"[{date_dir.name}] 통합 포트폴리오 {len(portfolio)}종목 저장")
        updated += 1

    logger.info(f"aggregator 재실행 완료: {updated}개 날짜")


def rerun_eval():
    collector = KoreanStockCollector()
    schedule_map = {p: e for p, e in MONTHLY_SCHEDULE + WEEKLY_SCHEDULE}

    for port_date, eval_date in sorted(schedule_map.items()):
        # 기존 eval 파일 삭제 → 재실행 강제
        eval_file = BACKTEST_DIR / port_date / f"eval_{eval_date}.json"
        if eval_file.exists():
            eval_file.unlink()
            logger.info(f"[{port_date}] 기존 eval 삭제")

        try:
            evaluate_portfolio(port_date, eval_date, collector)
        except Exception as e:
            logger.error(f"[{port_date}] 평가 실패: {e}")

    logger.info("eval 재실행 완료")


if __name__ == "__main__":
    logger.info("=== aggregator 재실행 (top_n=5) ===")
    rerun_aggregator()
    logger.info("=== eval 재실행 ===")
    rerun_eval()
    logger.info("=== 완료 ===")
