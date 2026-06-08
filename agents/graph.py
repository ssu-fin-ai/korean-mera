"""MERA LangGraph 그래프 정의

LangGraph StateGraph를 사용해 MERA 파이프라인을 DAG(방향성 비순환 그래프)로 구성한다.

전체 흐름:
  START
    → screener   (전체 종목 데이터 수집 + 전문가별 Top-N 후보 선별)
    → gate       (종목 패턴별 전문가 적합도 가중치 계산)
    → growth     ┐
    → value      │  (5개 전문가 노드 병렬 실행: 각자 최대 5종목 선택)
    → theme      │
    → dividend   │
    → crisis     ┘
    → aggregator (5개 전문가 picks 통합 → 복합 스코어 → TOP-N 포트폴리오)
    → reporter   (포트폴리오 → 텍스트 리포트 생성)
  END

LangGraph는 fan-out(_EXPERTS 리스트 → 각 expert 노드)과
fan-in(_EXPERTS 완료 시 aggregator 진입)을 자동으로 처리한다.
"""

from langgraph.graph import StateGraph, START, END

from agents.state import MERAState
from agents.screener import screener_node
from agents.gate_agent import gate_node
from agents.experts import (
    growth_node, value_node, theme_node, dividend_node, crisis_node,
)
from portfolio.aggregator import aggregator_node, reporter_node

# 5개 전문가 노드 이름 목록 (fan-out / fan-in 에지 정의에 재사용)
_EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]


def build_mera_graph():
    """컴파일된 MERA LangGraph 반환

    StateGraph에 노드와 에지를 등록한 뒤 compile()로 실행 가능한 그래프를 반환한다.
    반환된 그래프는 graph.invoke({"date": "...", "rag_label_key": "..."}) 로 실행한다.
    """
    graph = StateGraph(MERAState)

    # ── 노드 등록 ──────────────────────────────────────────────────
    graph.add_node("screener", screener_node)    # 데이터 수집 + 후보 선별
    graph.add_node("gate", gate_node)            # GateNet 가중치 계산
    graph.add_node("growth", growth_node)        # 성장주 전문가
    graph.add_node("value", value_node)          # 가치주 전문가
    graph.add_node("theme", theme_node)          # 테마주 전문가
    graph.add_node("dividend", dividend_node)    # 배당주 전문가
    graph.add_node("crisis", crisis_node)        # 위기종목 전문가
    graph.add_node("aggregator", aggregator_node)  # 신호 집계 + 포트폴리오 구성
    graph.add_node("reporter", reporter_node)    # 텍스트 리포트 생성

    # ── 에지 등록 (실행 순서) ─────────────────────────────────────
    graph.add_edge(START, "screener")
    graph.add_edge("screener", "gate")

    # gate → 전문가 5개 병렬 fan-out
    # LangGraph가 자동으로 5개 노드를 동시에 실행
    for expert in _EXPERTS:
        graph.add_edge("gate", expert)

    # 전문가 5개 완료 후 aggregator fan-in
    # 리스트를 전달하면 모든 선행 노드가 끝날 때까지 대기
    graph.add_edge(_EXPERTS, "aggregator")
    graph.add_edge("aggregator", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()
