"""포트폴리오 성과 평가: 실제 수익률 계산 + LLM 분석 → DB 저장

매주 실행되어 지난주 BUY 포트폴리오의 실제 성과를 검증한다.

평가 흐름:
  1. Supabase에서 평가 대상 포트폴리오 조회
  2. 종목별 실제 수익률 계산 (입력 날짜 종가 → 평가 날짜 종가)
  3. LLM으로 각 종목의 BUY 신호 적중 여부 + 오신호 원인 분석
  4. 포트폴리오 전체 요약 생성 (LLM)
  5. 결과를 Supabase에 저장 (중복 실행 시 덮어씀)

성과 지표: 평균수익률, 적중률, ARR(연환산), MDD, 소르티노, 칼마
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import math
import threading

import pandas as pd
from loguru import logger

from agents.base import call_llm, parse_json_response
from config import SETTINGS


def _to_dash(date_str: str) -> str:
    """YYYYMMDD 또는 YYYY-MM-DD → YYYY-MM-DD 정규화"""
    s = date_str.replace("-", "")
    return f"{s[:4]}-{s[4:6]}-{s[6:]}" if len(s) == 8 else date_str


def _to_plain(date_str: str) -> str:
    """YYYY-MM-DD → YYYYMMDD 정규화"""
    return date_str.replace("-", "")


def compute_actual_return(collector, ticker: str, port_date: str, eval_date: str) -> float | None:
    """포트폴리오 편입일 종가 → 평가일 종가 수익률 계산

    두 날짜 모두 YYYY-MM-DD 형식.
    각 날짜 이전 가장 가까운 거래일 종가를 사용 (공휴일 대응).
    수익률 = (평가일 종가 - 편입일 종가) / 편입일 종가
    """
    try:
        # port_date 이전 10일치까지 포함해 데이터 확보 (공휴일 연휴 대응)
        start = (
            datetime.strptime(_to_plain(port_date), "%Y%m%d") - timedelta(days=10)
        ).strftime("%Y%m%d")
        df = collector.get_ohlcv(ticker, start, _to_plain(eval_date))
        if df.empty:
            return None

        port_dt = pd.to_datetime(port_date)
        eval_dt = pd.to_datetime(eval_date)

        # 편입일 이전 마지막 거래일 종가 (매수 기준가)
        before = df[df.index <= port_dt]
        # 평가일 이전 마지막 거래일 종가 (매도 기준가)
        after = df[df.index <= eval_dt]
        if before.empty or after.empty:
            return None

        price_from = float(before.iloc[-1]["close"])
        price_to = float(after.iloc[-1]["close"])
        return (price_to - price_from) / price_from if price_from else None
    except Exception as e:
        logger.debug(f"{ticker} 수익률 계산 실패: {e}")
        return None


def evaluate_stock(
    ticker: str, name: str, signal_text: str,
    actual_ret: float, target_ret: float,
) -> dict:
    """LLM으로 개별 종목 BUY 신호 적중 여부 + 오신호 원인 + 교훈 분석

    naive_correct: LLM 없이 수익률 방향만으로 정오 판단 (LLM 파싱 실패 시 fallback).
    LLM은 신호 근거 텍스트까지 참고해 더 세밀한 분석 제공.
    """
    # 수익률 방향이 맞으면 naive 기준으로 정답 (예측이 양수면 실제도 양수여야 함)
    naive_correct = (actual_ret > 0) == (target_ret > 0) if target_ret != 0 else actual_ret > 0

    system = (
        "주식 투자 신호 평가 전문가. BUY 신호 대비 실제 주가를 평가. "
        "반드시 JSON만 응답."
    )
    user = (
        f"종목: {ticker} {name}\n"
        f"예측 수익률: {target_ret:+.1%} | 실제 수익률: {actual_ret:+.1%}\n"
        f"신호 근거:\n{signal_text}\n\n"
        "아래 JSON 형식으로 평가:\n```json\n"
        '{"correct": true, "miss_reason": "틀렸다면 이유(맞으면 빈문자열)", '
        '"lesson": "다음 신호 생성 시 반영할 교훈", "confidence_score": 0.8}\n```'
    )

    resp = call_llm(system, user)
    parsed = parse_json_response(resp)

    return {
        "ticker": ticker,
        "name": name,
        "actual_return": actual_ret,
        "target_return": target_ret,
        "correct": bool(parsed.get("correct", naive_correct)),
        "miss_reason": str(parsed.get("miss_reason", "")),
        "lesson": str(parsed.get("lesson", "")),
        "confidence_score": float(parsed.get("confidence_score", 0.5)),
    }


def _summarize_evaluation(
    portfolio_date: str, stock_evals: list[dict],
    avg_return: float, hit_rate: float,
) -> str:
    """LLM으로 포트폴리오 전체 성과 요약 생성 (3~5문장)

    최고수익·최대손실 종목과 오신호 원인을 함께 제시해
    다음 신호 생성 개선에 활용할 인사이트를 도출한다.
    """
    best = max(stock_evals, key=lambda x: x["actual_return"])
    worst = min(stock_evals, key=lambda x: x["actual_return"])
    wrong = [e for e in stock_evals if not e["correct"]]

    wrong_lines = "\n".join(
        f"- {e['ticker']} {e['name']}: {e['miss_reason']}" for e in wrong[:5]
    )

    system = "AI 투자 에이전트 성과 분석가. 한국어로 3~5문장으로 답하세요."
    user = (
        f"포트폴리오 날짜: {portfolio_date}\n"
        f"종목수:{len(stock_evals)} | 평균수익:{avg_return:+.1%} | 적중률:{hit_rate:.0%}\n"
        f"최고수익: {best['ticker']} {best['name']} {best['actual_return']:+.1%}\n"
        f"최대손실: {worst['ticker']} {worst['name']} {worst['actual_return']:+.1%}\n"
        f"오신호 원인:\n{wrong_lines or '없음'}\n\n"
        "포트폴리오 신호 품질을 평가하고 개선점을 제시하세요."
    )
    return call_llm(system, user)


def run_evaluation(portfolio_date: str, eval_date: str, collector) -> dict:
    """전일 포트폴리오 평가 실행 및 DB 저장

    이미 같은 eval_date로 평가된 경우 스킵 (멱등성).
    다른 eval_date로 재평가 시 기존 결과 덮어씀.
    portfolio_date / eval_date 모두 YYYY-MM-DD 형식.

    반환 dict 키: portfolio_date, eval_date, stock_count,
                  avg_return, hit_rate, arr, mdd, sortino, calmar, summary
    """
    from db.supabase_store import EvaluationStore, PortfolioStore

    eval_store = EvaluationStore()
    port_store = PortfolioStore()

    # 이미 동일 날짜로 평가된 경우 스킵
    existing = eval_store.get_summary_by_date(portfolio_date)
    if existing:
        existing_eval_date = existing.get("metadata", existing).get("eval_date", "")
        if existing_eval_date == eval_date:
            logger.info(f"이미 평가됨 ({portfolio_date} → {eval_date}) → 스킵")
            return {}
        logger.info(f"기존 평가 덮어씀: {portfolio_date} (이전 eval_date={existing_eval_date} → {eval_date})")

    entries = port_store.get_by_date(portfolio_date)
    if not entries:
        logger.info(f"평가할 포트폴리오 없음: {portfolio_date}")
        return {}

    logger.info(
        f"포트폴리오 평가 시작: {portfolio_date} → {eval_date} ({len(entries)}종목)"
    )

    # Claude API 동시 호출 수 제한 (settings.yaml pipeline.api_concurrency)
    pipe_cfg = SETTINGS.get("pipeline", {})
    api_sem = threading.Semaphore(pipe_cfg.get("api_concurrency", 5))

    def _eval_one(entry):
        """단일 종목 평가 (스레드풀 내 실행)"""
        # Supabase 응답: 평탄 dict / ChromaDB 응답: {"text":..., "metadata":{...}}
        meta = entry.get("metadata", entry)
        signal_text = entry.get("text") or entry.get("signal_text", "")
        actual_ret = compute_actual_return(
            collector, meta["ticker"], portfolio_date, eval_date
        )
        if actual_ret is None:
            return None
        # 세마포어로 동시 LLM 호출 수 제한
        with api_sem:
            return evaluate_stock(
                ticker=meta["ticker"],
                name=meta["name"],
                signal_text=signal_text,
                actual_ret=actual_ret,
                target_ret=float(meta.get("target_return", 0)),
            )

    # 최대 10개 스레드로 종목별 수익률 계산 + LLM 평가 병렬 처리
    stock_evals = []
    with ThreadPoolExecutor(max_workers=min(10, len(entries))) as pool:
        futures = [pool.submit(_eval_one, e) for e in entries]
        for f in as_completed(futures):
            try:
                result = f.result()
                if result:
                    stock_evals.append(result)
            except Exception as e:
                logger.warning(f"종목 평가 실패: {e}")

    if not stock_evals:
        logger.warning(f"평가 가능한 종목 없음: {portfolio_date}")
        return {}

    returns = [e["actual_return"] for e in stock_evals]
    avg_return = sum(returns) / len(returns)
    # 적중률: 예측 방향(양/음)이 실제와 일치한 비율
    hit_rate = sum(1 for e in stock_evals if e["correct"]) / len(stock_evals)

    # ── 리스크 지표 계산 (보유기간 5일 기준 연환산) ──────────────────────
    hold_days = SETTINGS.get("portfolio", {}).get("hold_days", 5)
    periods_per_year = 252 / hold_days  # 1년에 몇 번 포트폴리오를 회전하는지

    # ARR(연환산 수익률): (1 + avg_return)^periods_per_year - 1
    arr = float((1 + avg_return) ** periods_per_year - 1)

    # MDD(최대 낙폭): 고점 대비 최대 하락률
    cum, peak, mdd = 1.0, 1.0, 0.0
    for r in returns:
        cum *= (1 + r)
        peak = max(peak, cum)
        mdd = min(mdd, (cum - peak) / peak)  # 음수: 고점 대비 낙폭

    # 소르티노 비율: 하방 편차만 사용한 위험 대비 수익률
    downside = [r for r in returns if r < 0]
    if len(downside) >= 2:
        down_mean = sum(downside) / len(downside)
        down_std = math.sqrt(sum((r - down_mean) ** 2 for r in downside) / (len(downside) - 1))
        sortino = float(avg_return / down_std * math.sqrt(periods_per_year)) if down_std > 0 else 0.0
    else:
        sortino = 0.0

    # 칼마 비율: ARR / |MDD| (낙폭 대비 수익률)
    calmar = float(arr / abs(mdd)) if mdd != 0 else 0.0

    # LLM으로 전체 포트폴리오 성과 요약 생성
    summary = _summarize_evaluation(portfolio_date, stock_evals, avg_return, hit_rate)

    # Supabase에 평가 결과 저장
    eval_store.save_evaluation(
        portfolio_date=portfolio_date,
        eval_date=eval_date,
        stock_evals=stock_evals,
        summary=summary,
        avg_return=avg_return,
        hit_rate=hit_rate,
        arr=arr,
        mdd=mdd,
        sortino=sortino,
        calmar=calmar,
    )

    logger.info(
        f"평가 완료: 평균수익 {avg_return:+.1%} | 적중률 {hit_rate:.0%} | "
        f"ARR {arr:+.1%} | MDD {mdd:.1%} | 소르티노 {sortino:.2f} | {len(stock_evals)}종목"
    )
    return {
        "portfolio_date": portfolio_date,
        "eval_date": eval_date,
        "stock_count": len(stock_evals),
        "avg_return": avg_return,
        "hit_rate": hit_rate,
        "arr": arr,
        "mdd": mdd,
        "sortino": sortino,
        "calmar": calmar,
        "summary": summary,
    }
