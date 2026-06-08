"""LangGraph 상태(State) 정의

MERAState는 LangGraph 그래프 전체에서 공유되는 데이터 컨테이너다.
각 노드(screener → gate → experts → aggregator → reporter)가
이 상태를 읽고 쓰며 파이프라인이 진행된다.

데이터 흐름:
  screener_node    → growth/value/theme/dividend/crisis_candidates 채움
  gate_node        → gate_weights_map 채움
  expert_nodes     → growth/value/theme/dividend/crisis_picks 채움
  aggregator_node  → final_portfolio 채움
  reporter_node    → report 채움
"""
from typing import TypedDict


class ExpertPick(TypedDict):
    """전문가 에이전트가 선택한 개별 종목 정보"""
    ticker: str           # 종목 코드 (예: "005930")
    name: str             # 종목명 (예: "삼성전자")
    signal: str           # 투자 신호: "BUY" | "HOLD" | "SELL"
    confidence: float     # 신뢰도: 0.0 ~ 1.0 (높을수록 확신)
    score: int            # 매력도 점수: 1 ~ 10
    target_return: float  # 예상 수익률 (소수, 예: 0.10 = +10%)
    horizon_days: int     # 투자 기간 (일)
    reason: str           # 핵심 투자 근거 텍스트
    pros: list[str]       # 긍정 요인 목록 (구체적 수치 포함)
    cons: list[str]       # 리스크 요인 목록
    expert: str           # 담당 전문가: "growth"|"value"|"theme"|"dividend"|"crisis"


class MERAState(TypedDict):
    """LangGraph 전체 파이프라인 공유 상태"""

    # 분석 기준 날짜 (YYYYMMDD 형식)
    date: str

    # RAG 레이블 키: "label_5d" (주간 분석) | "label_20d" (월간 분석)
    # 유사 패턴 조회 시 어떤 기간의 수익률을 참조할지 결정
    rag_label_key: str

    # ── 스크리너 출력: 전문가별 후보 종목 리스트 ──────────────────
    # 각 리스트는 _collect_ticker()가 반환한 dict 목록
    # (ticker, name, sector, snapshot, financials, retrieved_patterns 등 포함)
    growth_candidates: list[dict]    # 성장주 후보 (실적 모멘텀, 고PER)
    value_candidates: list[dict]     # 가치주 후보 (저PBR/PER, 안정 현금흐름)
    theme_candidates: list[dict]     # 테마주 후보 (정책수혜, 이슈 모멘텀)
    dividend_candidates: list[dict]  # 배당주 후보 (고배당, 방어적 섹터)
    crisis_candidates: list[dict]    # 위기종목 후보 (급락 후 반등 가능성)

    # ── GateNet 출력: 종목별 전문가 적합도 가중치 ────────────────
    # 구조: {ticker: {growth: 0.8, value: 0.1, theme: 0.3, dividend: 0.0, crisis: 0.2}}
    # aggregator에서 복합 점수 계산 시 이 가중치를 곱해 전문가 의견 반영 비중 조정
    gate_weights_map: dict

    # ── 전문가 노드 출력: 각 전문가의 추천 종목 (최대 5개) ────────
    growth_picks: list[ExpertPick]
    value_picks: list[ExpertPick]
    theme_picks: list[ExpertPick]
    dividend_picks: list[ExpertPick]
    crisis_picks: list[ExpertPick]

    # ── 최종 결과 ──────────────────────────────────────────────────
    final_portfolio: list[dict]  # aggregator가 구성한 TOP-N BUY 종목 리스트
    report: str                  # reporter가 생성한 텍스트 리포트
