"""Supabase 기반 포트폴리오/평가 저장소

테이블:
  portfolio_history      - 일별 BUY 신호 종목
  portfolio_eval_summary - 포트폴리오 날짜별 성과 요약
  portfolio_eval_stocks  - 종목별 실제 수익률 + LLM 평가
"""

import json
import urllib.request
import urllib.error
from loguru import logger

from config import SUPABASE_URL, SUPABASE_KEY


def _rest(path: str) -> str:
    return f"{SUPABASE_URL}/rest/v1/{path}"


def _headers(prefer: str = "resolution=merge-duplicates") -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": prefer,
    }


def _pg_query(sql: str) -> list:
    """pg/query 엔드포인트로 원시 SQL 실행"""
    data = json.dumps({"query": sql}).encode()
    req = urllib.request.Request(
        f"{SUPABASE_URL}/pg/query",
        data=data,
        headers={
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _post(url: str, payload) -> tuple[int, bytes]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _upsert(table: str, payload, on_conflict: str) -> tuple[int, bytes]:
    """UNIQUE 충돌 시 UPDATE로 처리하는 upsert"""
    url = f"{_rest(table)}?on_conflict={on_conflict}"
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=_headers(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _get(url: str) -> tuple[int, list]:
    req = urllib.request.Request(url, headers=_headers(prefer=""), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, []


class PortfolioStore:
    """일별 BUY 신호 포트폴리오를 portfolio_history 테이블에 저장"""

    TABLE = "portfolio_history"

    def save(self, portfolio_date: str, portfolio_df) -> int:
        """포트폴리오 DataFrame → Supabase upsert"""
        rows = []
        for rank, row in portfolio_df.iterrows():
            reasons = row.get("reasons", [])
            reasons_text = "; ".join(reasons) if reasons else ""
            signal_text = (
                f"BUY 신호 {portfolio_date}: {row['ticker']} {row['name']} [{row['sector']}]\n"
                f"신뢰도:{row['confidence']:.0%} 점수:{row['score']:.2f} "
                f"목표수익:{row['target_return']:+.1%}\n"
                f"패턴:{row.get('pattern_type', '')}\n"
                f"근거:{reasons_text}"
            )
            rows.append({
                "portfolio_date": portfolio_date,
                "ticker": str(row["ticker"]),
                "name": str(row["name"]),
                "sector": str(row["sector"]),
                "signal": str(row["signal"]),
                "score": float(row["score"]),
                "confidence": float(row["confidence"]),
                "target_return": float(row["target_return"]),
                "pattern_type": str(row.get("pattern_type", "")),
                "rank": int(rank),
                "signal_text": signal_text,
            })

        if not rows:
            return 0

        status, body = _upsert(self.TABLE, rows, "portfolio_date,ticker")
        if status not in (200, 201):
            logger.error(f"PortfolioStore 저장 실패: {status} {body[:200]}")
            return 0

        logger.info(f"PortfolioStore: {len(rows)}종목 저장 ({portfolio_date})")
        return len(rows)

    def get_by_date(self, portfolio_date: str) -> list[dict]:
        """특정 날짜 포트폴리오 전체 조회"""
        status, data = _get(
            _rest(f"{self.TABLE}?portfolio_date=eq.{portfolio_date}&select=*&order=rank.asc")
        )
        if status != 200:
            logger.error(f"PortfolioStore 조회 실패: {status}")
            return []
        return data

    def list_dates(self) -> list[str]:
        """저장된 포트폴리오 날짜 목록 (오름차순)"""
        status, data = _get(
            _rest(f"{self.TABLE}?select=portfolio_date&order=portfolio_date.asc")
        )
        if status != 200:
            return []
        return sorted({r["portfolio_date"] for r in data})


class EvaluationStore:
    """포트폴리오 성과 평가 결과를 Supabase에 저장"""

    SUMMARY_TABLE = "portfolio_eval_summary"
    STOCK_TABLE = "portfolio_eval_stocks"

    def save_evaluation(
        self,
        portfolio_date: str,
        eval_date: str,
        stock_evals: list[dict],
        summary: str,
        avg_return: float,
        hit_rate: float,
        arr: float = 0.0,
        mdd: float = 0.0,
        sortino: float = 0.0,
        calmar: float = 0.0,
    ) -> None:
        # 요약 저장
        status, body = _upsert(
            self.SUMMARY_TABLE,
            {
                "portfolio_date": portfolio_date,
                "eval_date": eval_date,
                "stock_count": len(stock_evals),
                "avg_return": float(avg_return),
                "hit_rate": float(hit_rate),
                "arr": float(arr),
                "mdd": float(mdd),
                "sortino": float(sortino),
                "calmar": float(calmar),
                "summary_text": summary,
            },
            "portfolio_date",
        )
        if status not in (200, 201):
            logger.error(f"EvaluationStore 요약 저장 실패: {status} {body[:200]}")

        # 종목별 평가 저장
        if stock_evals:
            rows = [
                {
                    "portfolio_date": portfolio_date,
                    "eval_date": eval_date,
                    "ticker": str(e["ticker"]),
                    "name": str(e["name"]),
                    "actual_return": float(e["actual_return"]),
                    "target_return": float(e["target_return"]),
                    "correct": bool(e["correct"]),
                    "confidence_score": float(e.get("confidence_score", 0.5)),
                    "miss_reason": str(e.get("miss_reason", "")),
                    "lesson": str(e.get("lesson", "")),
                }
                for e in stock_evals
            ]
            status, body = _upsert(self.STOCK_TABLE, rows, "portfolio_date,ticker")
            if status not in (200, 201):
                logger.error(f"EvaluationStore 종목 저장 실패: {status} {body[:200]}")

        logger.info(
            f"EvaluationStore: {portfolio_date} 평가 저장 "
            f"(수익:{avg_return:+.1%} 적중:{hit_rate:.0%})"
        )

    def get_summary_by_date(self, portfolio_date: str) -> dict | None:
        """날짜별 요약 조회 (없으면 None)"""
        status, data = _get(
            _rest(f"{self.SUMMARY_TABLE}?portfolio_date=eq.{portfolio_date}&select=*")
        )
        if status != 200 or not data:
            return None
        return data[0]

    def get_stock_evals_by_date(self, portfolio_date: str) -> list[dict]:
        """날짜별 종목 평가 목록 조회"""
        status, data = _get(
            _rest(f"{self.STOCK_TABLE}?portfolio_date=eq.{portfolio_date}&select=*")
        )
        return data if status == 200 else []

    def get_recent_summaries(self, n: int = 10) -> list[dict]:
        """최근 N개 요약 (날짜 내림차순)"""
        status, data = _get(
            _rest(f"{self.SUMMARY_TABLE}?select=*&order=portfolio_date.desc&limit={n}")
        )
        return data if status == 200 else []


def init_tables() -> None:
    """테이블이 없으면 생성 (멱등)"""
    ddl_list = [
        """CREATE TABLE IF NOT EXISTS portfolio_history (
            id BIGSERIAL PRIMARY KEY,
            portfolio_date DATE NOT NULL,
            ticker VARCHAR(10) NOT NULL,
            name VARCHAR(200), sector VARCHAR(200), signal VARCHAR(10),
            score FLOAT8, confidence FLOAT8, target_return FLOAT8,
            pattern_type TEXT, rank INTEGER, signal_text TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT portfolio_history_uq UNIQUE(portfolio_date, ticker)
        )""",
        """CREATE TABLE IF NOT EXISTS portfolio_eval_summary (
            id BIGSERIAL PRIMARY KEY,
            portfolio_date DATE NOT NULL UNIQUE,
            eval_date DATE NOT NULL,
            stock_count INTEGER, avg_return FLOAT8, hit_rate FLOAT8,
            arr FLOAT8, mdd FLOAT8, sortino FLOAT8, calmar FLOAT8,
            summary_text TEXT, created_at TIMESTAMPTZ DEFAULT NOW()
        )""",
        """CREATE TABLE IF NOT EXISTS portfolio_eval_stocks (
            id BIGSERIAL PRIMARY KEY,
            portfolio_date DATE NOT NULL, eval_date DATE NOT NULL,
            ticker VARCHAR(10) NOT NULL, name VARCHAR(200),
            actual_return FLOAT8, target_return FLOAT8, correct BOOLEAN,
            confidence_score FLOAT8, miss_reason TEXT, lesson TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            CONSTRAINT portfolio_eval_stocks_uq UNIQUE(portfolio_date, ticker)
        )""",
    ]
    for ddl in ddl_list:
        try:
            _pg_query(ddl)
        except Exception as e:
            logger.warning(f"테이블 생성 중 오류 (무시): {e}")
    logger.info("Supabase 테이블 초기화 완료")
