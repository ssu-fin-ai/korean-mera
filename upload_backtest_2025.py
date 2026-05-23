"""로컬 reports/backtest_2025/ JSON → Supabase 업로드

실행:
    py upload_backtest_2025.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
from loguru import logger

from config import ROOT
from db.supabase_store import BacktestStore, init_tables, migrate_eval_constraints
from backtest_2025 import MONTHLY_SCHEDULE

BACKTEST_DIR = ROOT / "reports" / "backtest_2025"
EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]


def upload():
    logger.info("Supabase 테이블 초기화...")
    init_tables()

    store = BacktestStore()
    total_picks = 0
    total_evals = 0

    # ── picks: 날짜 디렉터리 기준 ──
    for date_dir in sorted(BACKTEST_DIR.glob("2025*")):
        port_date = date_dir.name
        for expert in EXPERTS + ["combined"]:
            fname = f"{expert}_picks.json" if expert != "combined" else "combined_portfolio.json"
            f = date_dir / fname
            if not f.exists():
                continue
            picks = json.loads(f.read_text(encoding="utf-8"))
            saved = store.save_picks(port_date, expert, picks)
            total_picks += saved
            if saved:
                logger.info(f"  [{port_date}] {expert}: {saved}종목 picks 저장")

    # ── eval: 월별 순회 ──
    for port_date, eval_date in MONTHLY_SCHEDULE:
        eval_file = BACKTEST_DIR / port_date / f"eval_{eval_date}.json"
        if not eval_file.exists():
            logger.warning(f"  [{port_date}->{eval_date}] eval 파일 없음, 스킵")
            continue
        results = json.loads(eval_file.read_text(encoding="utf-8"))
        store.save_eval(port_date, eval_date, results)
        total_evals += 1

    logger.info(f"=== 업로드 완료: picks {total_picks}건, eval {total_evals}회 ===")


if __name__ == "__main__":
    upload()
