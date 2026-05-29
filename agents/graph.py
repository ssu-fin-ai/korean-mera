"""MERA LangGraph 그래프

흐름:
  START → screener → gate → [growth, value, theme, dividend, crisis] (병렬) → aggregator → reporter → END
"""

from langgraph.graph import StateGraph, START, END

from agents.state import MERAState
from agents.screener import screener_node
from agents.gate_agent import gate_node
from agents.experts import (
    growth_node, value_node, theme_node, dividend_node, crisis_node,
)
from portfolio.aggregator import aggregator_node, reporter_node

_EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]


def build_mera_graph():
    """컴파일된 MERA LangGraph 반환"""
    graph = StateGraph(MERAState)

    graph.add_node("screener", screener_node)
    graph.add_node("gate", gate_node)
    graph.add_node("growth", growth_node)
    graph.add_node("value", value_node)
    graph.add_node("theme", theme_node)
    graph.add_node("dividend", dividend_node)
    graph.add_node("crisis", crisis_node)
    graph.add_node("aggregator", aggregator_node)
    graph.add_node("reporter", reporter_node)

    graph.add_edge(START, "screener")
    graph.add_edge("screener", "gate")

    # gate → 전문가 5개 병렬 fan-out
    for expert in _EXPERTS:
        graph.add_edge("gate", expert)

    # 전문가 5개 완료 후 aggregator fan-in
    graph.add_edge(_EXPERTS, "aggregator")
    graph.add_edge("aggregator", "reporter")
    graph.add_edge("reporter", END)

    return graph.compile()
