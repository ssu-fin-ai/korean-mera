"""LangGraph 상태 정의"""
from typing import TypedDict


class ExpertPick(TypedDict):
    ticker: str
    name: str
    signal: str
    confidence: float
    score: int
    target_return: float
    horizon_days: int
    reason: str
    pros: list[str]
    cons: list[str]
    expert: str


class MERAState(TypedDict):
    date: str
    rag_label_key: str  # "label_5d" (weekly) | "label_20d" (monthly)
    # 전문가별 후보 종목 (스크리닝 결과)
    growth_candidates: list[dict]
    value_candidates: list[dict]
    theme_candidates: list[dict]
    dividend_candidates: list[dict]
    crisis_candidates: list[dict]
    # GateNet 전문가 가중치 {ticker: {growth: 0.8, value: 0.1, ...}}
    gate_weights_map: dict
    # 전문가별 추천 종목
    growth_picks: list[ExpertPick]
    value_picks: list[ExpertPick]
    theme_picks: list[ExpertPick]
    dividend_picks: list[ExpertPick]
    crisis_picks: list[ExpertPick]
    # 최종 결과
    final_portfolio: list[dict]
    report: str
