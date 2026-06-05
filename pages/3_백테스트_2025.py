"""2025년 월별 백테스트 대시보드 (12개월)"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from config import ROOT
from db.supabase_store import BacktestStore

BACKTEST_DIR = ROOT / "reports" / "backtest_2025"

_store = BacktestStore()

MONTHLY_SCHEDULE = [
    ("20250102", "20250131"),
    ("20250203", "20250228"),
    ("20250303", "20250331"),
    ("20250401", "20250430"),
    ("20250502", "20250530"),
    ("20250602", "20250630"),
    ("20250701", "20250731"),
    ("20250801", "20250829"),
    ("20250901", "20250930"),
    ("20251001", "20251031"),
    ("20251103", "20251128"),
    ("20251201", "20251231"),
]

MONTH_LABELS = {
    "01": "01월", "02": "02월", "03": "03월", "04": "04월",
    "05": "05월", "06": "06월", "07": "07월", "08": "08월",
    "09": "09월", "10": "10월", "11": "11월", "12": "12월",
}

EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]
EXPERT_KOR = {
    "growth": "성장주", "value": "가치주", "theme": "테마주",
    "dividend": "배당주", "crisis": "위기종목", "combined": "통합",
}


# ── 유틸 ─────────────────────────────────────────────────────────

def _dash(d): return f"{d[:4]}-{d[4:6]}-{d[6:]}"
def _to_dash(d): return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and "-" not in d else d
def _fmt(v, f="+.1%"): return f"{v:{f}}" if v is not None else "-"
def _month_label(port_date): return MONTH_LABELS.get(port_date[4:6], port_date)


# ── 데이터 로더 ──────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_picks(port_date, expert):
    rows = _store.get_picks(_to_dash(port_date), expert)
    if rows:
        return rows
    fname = f"{expert}_picks.json" if expert != "combined" else "combined_portfolio.json"
    f = BACKTEST_DIR / port_date / fname
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

@st.cache_data(ttl=300)
def load_eval(port_date, eval_date):
    result = _store.get_eval(_to_dash(port_date), _to_dash(eval_date))
    if result:
        return result
    f = BACKTEST_DIR / port_date / f"eval_{eval_date}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


# ── 지표 계산 ─────────────────────────────────────────────────────

def calc_metrics(rets: list[float]) -> dict:
    if not rets:
        return {"cum": 0.0, "arr": 0.0, "mdd": 0.0, "sortino": 0.0, "calmar": 0.0, "hit": 0.0}
    r = np.array(rets)
    ann = 252 / 20  # 월별 보유 ~20일
    cum_r = np.cumprod(1 + r)
    cum = float(cum_r[-1] - 1)
    arr = float((1 + cum) ** (ann / len(r)) - 1)
    peak = np.maximum.accumulate(cum_r)
    mdd = float(((cum_r - peak) / peak).min())
    down = r[r < 0]
    d_std = float(down.std()) if len(down) > 1 else 1e-6
    sortino = float((r.mean() / d_std) * np.sqrt(ann))
    calmar = arr / abs(mdd) if mdd != 0 else 0.0
    hit = float(np.mean(r > 0))
    return {"cum": cum, "arr": arr, "mdd": mdd, "sortino": sortino, "calmar": calmar, "hit": hit}

def get_expert_rets(expert):
    return [r for r in
            (load_eval(p, e).get(expert, {}).get("avg_return")
             for p, e in MONTHLY_SCHEDULE if load_eval(p, e).get(expert))
            if r is not None]

def build_summary_df():
    rows = []
    for port_date, eval_date in MONTHLY_SCHEDULE:
        ev = load_eval(port_date, eval_date)
        for expert in EXPERTS + ["combined"]:
            if expert not in ev:
                continue
            r = ev[expert]
            rows.append({
                "월": _month_label(port_date),
                "port_date": port_date,
                "전문가": EXPERT_KOR.get(expert, expert),
                "expert_key": expert,
                "평균수익": r["avg_return"],
                "적중률": r["hit_rate"],
                "종목수": r["stock_count"],
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 페이지
# ════════════════════════════════════════════════════════════════

st.title("📅 MERA 2025 백테스트")
st.caption("2025년 1월~12월 월별 포트폴리오 성과 | KOSPI200 | 보유 ~20일")

section = st.radio(
    "섹션",
    ["📊 연간 성과 요약", "📋 월별 상세 리포트", "🗂 포트폴리오 조회", "📖 전문가 분석 방법론"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()


# ════════════════════════════════════════════════════════════════
# 섹션 1: 연간 성과 요약
# ════════════════════════════════════════════════════════════════

if section == "📊 연간 성과 요약":
    st.subheader("📊 2025년 전문가별 연간 누적 성과")

    df_sum = build_summary_df()

    # 전문가 카드
    cols = st.columns(len(EXPERTS))
    for col, expert in zip(cols, EXPERTS):
        rets = get_expert_rets(expert)
        m = calc_metrics(rets)
        col.metric(
            EXPERT_KOR[expert],
            f"누적 {m['cum']:+.1%}",
            delta=f"ARR {m['arr']:+.1%}",
            delta_color="normal" if m["arr"] >= 0 else "inverse",
        )

    # 통합 카드
    comb_rets = get_expert_rets("combined")
    comb_m = calc_metrics(comb_rets)
    st.divider()
    st.markdown("#### 🎯 통합 포트폴리오")
    ca, cb, cc, cd, ce = st.columns(5)
    ca.metric("누적수익", _fmt(comb_m["cum"]))
    cb.metric("ARR (연환산)", _fmt(comb_m["arr"]))
    cc.metric("MDD", f"{comb_m['mdd']:.1%}")
    cd.metric("Sortino", f"{comb_m['sortino']:.2f}")
    ce.metric("평균 적중률", f"{comb_m['hit']:.0%}")

    st.divider()

    # 리스크 지표 테이블
    st.subheader("전문가별 리스크 지표 (12개월)")

    def _cr(val):
        try:
            v = float(str(val).replace("%", "").replace("+", ""))
            return f"color:{'#2ecc71' if v > 0 else '#e74c3c'};font-weight:bold"
        except:
            return ""

    risk_rows = []
    for expert in EXPERTS + ["combined"]:
        rets = get_expert_rets(expert)
        m = calc_metrics(rets)
        risk_rows.append({
            "전문가": EXPERT_KOR.get(expert, expert),
            "누적수익 (12회)": _fmt(m["cum"]),
            "ARR (연환산)": _fmt(m["arr"]),
            "MDD": f"{m['mdd']:.1%}",
            "Sortino": f"{m['sortino']:.2f}",
            "Calmar": f"{m['calmar']:.2f}",
            "평균 적중률": f"{m['hit']:.0%}",
        })

    st.dataframe(
        pd.DataFrame(risk_rows).style.map(_cr, subset=["누적수익 (12회)", "ARR (연환산)"]),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # 월별 수익률 추이 차트
    st.subheader("월별 평균수익률 추이")
    if not df_sum.empty and "expert_key" in df_sum.columns:
        pivot = (
            df_sum[df_sum["expert_key"].isin(EXPERTS)]
            .pivot(index="port_date", columns="전문가", values="평균수익")
            .reindex([p for p, _ in MONTHLY_SCHEDULE])
        )
        month_order = [f"{i:02d}월" for i in range(1, 13)]
        pivot.index = month_order[:len(pivot)]
        pivot.index.name = "월"
        pivot_long = pivot.reset_index().melt(id_vars="월", var_name="전문가", value_name="평균수익")
        chart = (
            alt.Chart(pivot_long).mark_bar().encode(
                x=alt.X("월:O", sort=month_order, title=""),
                y=alt.Y("평균수익:Q", axis=alt.Axis(format=".0%", title="평균수익률")),
                color=alt.Color("전문가:N"),
                xOffset="전문가:N",
            ).properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("평가 데이터가 없습니다.")

    # 최고/최저 월 하이라이트
    st.divider()
    st.subheader("📌 월별 하이라이트")
    month_avgs = []
    for port_date, eval_date in MONTHLY_SCHEDULE:
        ev = load_eval(port_date, eval_date)
        all_rets = [ev[e]["avg_return"] for e in EXPERTS if e in ev]
        if all_rets:
            month_avgs.append((_month_label(port_date), sum(all_rets) / len(all_rets)))

    if month_avgs:
        best = max(month_avgs, key=lambda x: x[1])
        worst = min(month_avgs, key=lambda x: x[1])
        h1, h2, h3 = st.columns(3)
        h1.success(f"🏆 **최고 月**: {best[0]}  {best[1]:+.1%}")
        h2.error(f"📉 **최저 月**: {worst[0]}  {worst[1]:+.1%}")
        avg_all = sum(v for _, v in month_avgs) / len(month_avgs)
        h3.info(f"📊 **전체 평균**: {avg_all:+.1%}")


# ════════════════════════════════════════════════════════════════
# 섹션 2: 월별 상세 리포트
# ════════════════════════════════════════════════════════════════

elif section == "📋 월별 상세 리포트":
    st.subheader("📋 월별 상세 성과")

    eval_map = dict(MONTHLY_SCHEDULE)
    date_labels = {_month_label(p): p for p, _ in MONTHLY_SCHEDULE}
    sel_label = st.selectbox("📅 월 선택", list(date_labels.keys()))
    sel_port = date_labels[sel_label]
    sel_eval = eval_map[sel_port]
    ev = load_eval(sel_port, sel_eval)

    st.caption(f"포트폴리오: {_dash(sel_port)}  →  평가일: {_dash(sel_eval)}")

    if not ev:
        st.info("평가 데이터 없음")
    else:
        # 전문가 지표 카드
        mcols = st.columns(len(EXPERTS))
        for col, exp in zip(mcols, EXPERTS):
            if exp not in ev:
                col.metric(EXPERT_KOR[exp], "-")
                continue
            r = ev[exp]
            ret = r["avg_return"]
            col.metric(
                EXPERT_KOR[exp],
                _fmt(ret),
                delta=f"적중 {r['hit_rate']:.0%} / {r['stock_count']}종목",
                delta_color="normal" if ret >= 0 else "inverse",
            )

        st.divider()

        def _sr(val):
            try:
                v = float(str(val).replace("%", "").replace("+", ""))
                return f"color:{'#2ecc71' if v > 0 else '#e74c3c'};font-weight:bold"
            except:
                return ""

        # 전문가별 종목 실적
        for exp in EXPERTS + ["combined"]:
            if exp not in ev:
                continue
            r = ev[exp]
            ret = r["avg_return"]
            label = EXPERT_KOR.get(exp, exp)
            st.markdown(
                f"#### {'📈' if ret >= 0 else '📉'} {label}"
                f"　평균수익: **{_fmt(ret)}**"
                f"　적중률: **{r['hit_rate']:.0%}**"
                f"　({r['stock_count']}종목)"
            )
            stocks = sorted(r["stocks"], key=lambda x: x["actual_return"], reverse=True)
            rows = [
                {
                    "티커": s["ticker"], "종목명": s["name"],
                    "목표수익": s["target_return"], "실제수익": s["actual_return"],
                    "적중": "✅" if s["hit"] else "❌",
                    "선정이유": s.get("reason", ""),
                }
                for s in stocks
            ]
            df_s = pd.DataFrame(rows)
            cc, ct = st.columns([1, 2], gap="medium")
            with cc:
                st.bar_chart(df_s.set_index("종목명")[["목표수익", "실제수익"]], height=200)
            with ct:
                d2 = df_s.copy()
                d2["목표수익"] = d2["목표수익"].apply(lambda x: _fmt(x))
                d2["실제수익"] = d2["실제수익"].apply(lambda x: _fmt(x))
                st.dataframe(d2.style.map(_sr, subset=["실제수익"]),
                             use_container_width=True, hide_index=True)
            st.divider()


# ════════════════════════════════════════════════════════════════
# 섹션 3: 포트폴리오 조회
# ════════════════════════════════════════════════════════════════

elif section == "🗂 포트폴리오 조회":
    st.subheader("🗂 선정 종목 조회")

    eval_map = dict(MONTHLY_SCHEDULE)
    date_labels = {_month_label(p): p for p, _ in MONTHLY_SCHEDULE}

    c1, c2 = st.columns(2)
    sel_label = c1.selectbox("📅 월", list(date_labels.keys()))
    sel_expert = c2.selectbox("전문가", [EXPERT_KOR[e] for e in EXPERTS] + ["통합"])
    expert_key_map = {v: k for k, v in EXPERT_KOR.items()}

    sel_port = date_labels[sel_label]
    sel_eval = eval_map[sel_port]
    sel_exp = expert_key_map.get(sel_expert, "combined")

    ev_port = load_eval(sel_port, sel_eval)
    picks = load_picks(sel_port, sel_exp)

    st.caption(f"포트폴리오: {_dash(sel_port)}  →  평가일: {_dash(sel_eval)}")

    if not picks:
        st.info("선정 종목 없음")
    else:
        buy = [p for p in picks if p.get("signal") == "BUY"] or picks
        ev_data = ev_port.get(sel_exp)
        ev_map = {s["ticker"]: s for s in (ev_data.get("stocks") if ev_data else [])}

        if ev_data:
            m1, m2, m3 = st.columns(3)
            ret = ev_data.get("avg_return", 0)
            m1.metric("평균수익률", _fmt(ret), delta_color="normal" if ret >= 0 else "inverse")
            m2.metric("적중률", f"{ev_data.get('hit_rate', 0):.0%}")
            m3.metric("종목수", f"{ev_data.get('stock_count', len(buy))}개")

        rows = []
        for i, p in enumerate(buy, 1):
            t = p.get("ticker", "")
            ev = ev_map.get(t, {})
            rows.append({
                "#": i, "티커": t, "종목명": p.get("name", ""),
                "신뢰도": f"{p.get('confidence', 0):.0%}",
                "목표수익": _fmt(p.get("target_return") or 0),
                "실제수익": _fmt(ev.get("actual_return")) if ev.get("actual_return") is not None else "-",
                "적중": ("✅" if ev["hit"] else "❌") if "hit" in ev else "-",
            })

        def _sc(val):
            if val == "-": return ""
            try:
                v = float(str(val).replace("%", "").replace("+", ""))
                return f"color:{'#2ecc71' if v > 0 else '#e74c3c'};font-weight:bold"
            except:
                return ""

        st.dataframe(
            pd.DataFrame(rows).style.map(_sc, subset=["실제수익"]),
            use_container_width=True, hide_index=True,
        )

        with st.expander("선정이유 / 긍정 / 리스크"):
            for p in buy:
                st.markdown(f"**{p.get('ticker', '')} {p.get('name', '')}**")
                if p.get("reason"): st.caption(f"📌 {p['reason']}")
                c1, c2 = st.columns(2)
                pros = " / ".join((p.get("pros") or [])[:2])
                cons = " / ".join((p.get("cons") or [])[:2])
                if pros: c1.markdown(f"✅ {pros}")
                if cons: c2.markdown(f"⚠️ {cons}")
                st.markdown("---")


# ════════════════════════════════════════════════════════════════
# 섹션 4: 전문가 분석 방법론
# ════════════════════════════════════════════════════════════════

elif section == "📖 전문가 분석 방법론":
    st.subheader("📖 전문가 분석 방법론")
    st.caption("MERA 에이전트가 종목을 선정하는 전체 프로세스와 각 전문가의 분석 기준을 설명합니다.")

    st.markdown("### 🔄 전체 종목 선정 프로세스")
    st.markdown("""
| 단계 | 이름 | 설명 |
|------|------|------|
| 1 | **스크리너** | KOSPI200 전체 종목 데이터 수집 → 전문가별 휴리스틱 점수로 Top-15 후보 필터링 |
| 2 | **전문가 에이전트** | 각 전문가가 Top-15 후보를 LLM(Claude)으로 심층 분석 → 최대 5종목 선택 |
| 3 | **어그리게이터** | 5개 전문가의 선정 결과 통합 → 신뢰도 가중 평균으로 최종 TOP-5 구성 |
| 4 | **평가** | 보유 기간 후 실제 수익률 측정 → 적중률·ARR·MDD·Sortino 산출 |
""")
    st.info(
        "**LLM 모델**: Claude Sonnet 4.6 (Anthropic API)  |  "
        "**임베딩**: OpenAI text-embedding-3-small (유사 패턴 검색)  |  "
        "**병렬 처리**: 전문가 5명 동시 실행 (LangGraph fan-out)"
    )
    st.divider()

    st.markdown("### 1️⃣ 스크리너 — 사전 후보 필터링")
    st.markdown(
        "전체 종목에 대해 전문가별 **휴리스틱 점수**를 계산하고, 점수가 높은 **상위 15종목**을 "
        "각 전문가에게 전달합니다. 수집 항목은 OHLCV, 기술적 지표, 재무제표, 공매도 비율, "
        "DART 공시, 섹터 평균 대비 밸류에이션입니다."
    )
    screener_cols = st.columns(5)
    screener_data = [
        ("📈 성장주", [
            ("거래량비율", "평균 대비 거래량, 최대 2점"),
            ("MACD", "양전환 시 +1점"),
            ("RSI", "45~72 구간 +1점"),
            ("5일 수익률", "상승폭 비례, 최대 2점"),
            ("영업이익 YoY", "성장률 비례, 최대 2점"),
            ("매출 YoY", "성장률 비례, 최대 1점"),
        ]),
        ("💰 가치주", [
            ("PBR", "0 < PBR < 1.5 구간 점수"),
            ("PER", "0 < PER < 15 구간 점수"),
            ("섹터 대비 PER", "섹터보다 낮을수록 가산"),
            ("섹터 대비 PBR", "섹터보다 낮을수록 가산"),
            ("ROE", "> 5% 시 최대 1점"),
            ("부채비율", "< 100% 시 +1점"),
        ]),
        ("🔥 테마주", [
            ("거래량비율", "> 3x +3점, > 2x +2점, > 1.5x +1점"),
            ("5일 수익률", "> 5% +2점, > 2% +1점"),
            ("RSI", "55~75 구간 +1점"),
            ("공매도비율", "> 5% +0.5점 (역매매)"),
        ]),
        ("💵 배당주", [
            ("배당수익률", "> 4% +3점, > 2% +1점"),
            ("배당성향", "0 < 70% 이하 +1점"),
            ("FCF", "양수 시 +2점"),
            ("베타", "0 < 0.8 (저변동) +1점"),
            ("부채비율", "< 100% +1점"),
        ]),
        ("🚨 위기종목", [
            ("5일 수익률", "< -10% +3점, < -5% +1점"),
            ("RSI", "< 30 +3점, < 40 +1점"),
            ("BB 위치", "하단 10% 이하 +2점"),
            ("부채비율", "안전 구간 +1점"),
            ("유동비율", "> 150% +1점"),
        ]),
    ]
    for col, (title, items) in zip(screener_cols, screener_data):
        col.markdown(f"**{title}**")
        for name, desc in items:
            col.markdown(f"- **{name}**: {desc}")
    st.divider()

    st.markdown("### 2️⃣ 전문가 에이전트 — LLM 심층 분석")
    st.markdown("스크리너가 넘긴 Top-15 후보를 LLM이 분석하여 **최대 5종목**을 선정합니다.")
    experts_info = [
        {
            "icon": "📈", "name": "성장주 전문가", "key": "growth",
            "role": "고PER·실적 모멘텀 종목 전문 애널리스트",
            "criteria": [
                ("실적 모멘텀", "매출·영업이익 YoY 성장률이 높은 순서로 평가"),
                ("주가 모멘텀", "RSI 45~70 (과열 제외), MACD 양전환, 거래량 동반 상승"),
                ("섹터 성장성", "AI·반도체·2차전지·바이오 등 고성장 섹터 우선"),
                ("유사 패턴", "OpenAI 임베딩으로 검색한 과거 유사 패턴의 5일 수익률 참조"),
            ],
            "data": "현재가·시총·PER·EPS·ROE·매출/영업이익/순이익 YoY·RSI·MACD·거래량비율·공매도비율",
            "horizon": "중기 (10~20일)",
        },
        {
            "icon": "💰", "name": "가치주 전문가", "key": "value",
            "role": "저PBR/PER·자산가치·안정 현금흐름 전문 애널리스트",
            "criteria": [
                ("밸류에이션", "PBR < 1.5, 섹터 대비 저PER 종목 우선"),
                ("Graham Number", "그레이엄 넘버 대비 현재가가 낮을수록 매력적"),
                ("재무 안정성", "부채비율·이자보상배율로 파산 리스크 점검"),
                ("저평가 해소 촉매", "배당수익률·FCF 등 주주환원 여력 확인"),
            ],
            "data": "현재가·PER·PBR·EPS·BPS·섹터 대비 PER/PBR·Graham Number·ROE·ROA·부채비율·유동비율·이자보상·배당·FCF·52주 위치",
            "horizon": "중장기 (20~30일)",
        },
        {
            "icon": "🔥", "name": "테마주 전문가", "key": "theme",
            "role": "정책 수혜·이슈 모멘텀·단기 급등 전문 애널리스트",
            "criteria": [
                ("공시·정책 수혜", "DART 최근 30일 공시 분석, 정책 수혜 직접성 평가"),
                ("거래량 + 모멘텀", "거래량 폭발적 증가 + RSI 70 미만 (과열 배제)"),
                ("테마 사이클", "초기 진입 우선, 말기(거품) 종목 배제"),
                ("유사 급등 패턴", "과거 유사 급등 후 수익률 패턴으로 지속성 판단"),
            ],
            "data": "현재가·RSI·MACD·거래량비율·5/20일 수익률·공매도비율·52주 고점 대비·DART 공시 요약",
            "horizon": "단기 (5~10일)",
        },
        {
            "icon": "💵", "name": "배당주 전문가", "key": "dividend",
            "role": "고배당·방어적 섹터·안정 수익 전문 애널리스트",
            "criteria": [
                ("배당 지속 가능성", "배당수익률 + FCF 기반 지속 가능성, 배당성향 30~70% 적정"),
                ("방어적 특성", "낮은 베타(< 0.8)·역사적 변동성으로 하방 안정성 평가"),
                ("재무 건전성", "부채비율·이자보상배율로 배당 유지 능력 확인"),
                ("금리 대비 매력도", "현재 금리 수준 대비 배당수익률의 상대적 매력 판단"),
            ],
            "data": "현재가·배당수익률·DPS·배당성향·FCF·OCF·ROE·ROA·부채비율·이자보상·베타·역사적변동성·5/20일 수익률",
            "horizon": "중장기 (20~30일)",
        },
        {
            "icon": "🚨", "name": "위기종목 전문가", "key": "crisis",
            "role": "급락 후 반등 가능성 전문 애널리스트",
            "criteria": [
                ("급락 원인 분석", "일시적 악재(실적 쇼크, 수급 이탈) vs 구조적 문제(사업 붕괴) 구분"),
                ("기술적 과매도", "RSI 30 이하, 볼린저밴드 하단 접근으로 매도 탈진 확인"),
                ("재무 건전성", "이자보상배율·유동비율로 반등 지속 가능성 확인"),
                ("역사적 회복 패턴", "유사 급락 후 OpenAI 임베딩 패턴 검색으로 과거 회복률 참조"),
            ],
            "data": "현재가·1/5/20일 수익률·RSI·BB 위치·거래량비율·52주 고/저점 대비·공매도비율·부채비율·유동비율·이자보상·DART 공시",
            "horizon": "단기~중기 (5~15일)",
        },
    ]
    for exp in experts_info:
        with st.expander(f"{exp['icon']} **{exp['name']}** — {exp['role']}", expanded=False):
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown("**선정 기준 (LLM 분석 가이드)**")
                for i, (title, desc) in enumerate(exp["criteria"], 1):
                    st.markdown(f"{i}. **{title}**: {desc}")
                st.markdown(f"\n**투자 기간**: {exp['horizon']}")
            with c2:
                st.markdown("**LLM에 제공되는 데이터**")
                st.caption(exp["data"])
                st.markdown("**출력 형식**")
                st.code(
                    '{\n'
                    '  "picks": [{\n'
                    '    "ticker": "005930",\n'
                    '    "signal": "BUY",\n'
                    '    "confidence": 0.85,\n'
                    '    "target_return": 0.10,\n'
                    '    "horizon_days": 10,\n'
                    '    "score": 8,\n'
                    '    "reason": "핵심 근거",\n'
                    '    "pros": ["긍정요인1", ...],\n'
                    '    "cons": ["리스크1", ...]\n'
                    '  }]\n'
                    '}',
                    language="json",
                )
    st.divider()

    st.markdown("### 3️⃣ 주요 기술적 지표 설명")
    indicator_data = {
        "지표": ["RSI", "MACD", "볼린저밴드 위치", "거래량비율", "베타", "YoY 성장률", "Graham Number", "공매도비율"],
        "설명": [
            "상대강도지수 (0~100). 70 이상 과매수, 30 이하 과매도",
            "단기·장기 이동평균 차이. 양전환(+)이면 상승 모멘텀 신호",
            "현재가가 볼린저밴드 내 위치 (0=하단, 1=상단). 0.1 이하 과매도",
            "최근 거래량 ÷ 20일 평균. 2x 이상이면 이슈 발생 신호",
            "KOSPI 대비 민감도. 1이면 시장과 동일, 0.5면 절반 변동",
            "전년 동기 대비 성장률. 매출/영업이익/순이익 기준",
            "√(EPS × BPS × 22.5). 벤저민 그레이엄 적정가치 기준",
            "전체 거래 대비 공매도 비중(%). 높을수록 하락 베팅 많음",
        ],
        "전문가 활용": [
            "성장: 45~70 / 테마: 55~75 / 위기: 30이하",
            "성장·테마: 양전환 확인",
            "위기: 0.1~0.2 이하 반등 신호",
            "성장·테마: 2x 이상 / 위기: 급등 주의",
            "배당: 0.8 이하 방어주 선호",
            "성장: 높을수록 / 가치: 재무 안정성 확인",
            "가치: 현재가/Graham 비율 낮을수록 저평가",
            "테마: 역매매 신호 / 위기: 숏 커버링 반등 가능성",
        ],
    }
    st.dataframe(pd.DataFrame(indicator_data), use_container_width=True, hide_index=True)
    st.divider()

    st.markdown("### 4️⃣ 최종 포트폴리오 구성 기준")
    st.markdown("""
5개 전문가가 각각 최대 5종목을 선정한 뒤, 어그리게이터가 다음 기준으로 최종 포트폴리오를 구성합니다.

| 기준 | 내용 |
|------|------|
| **신뢰도 임계값** | confidence ≥ 0.60인 BUY 신호만 채택 |
| **중복 가산점** | 여러 전문가가 동일 종목 추천 시 점수 합산 |
| **최종 종목 수** | 신뢰도 가중 점수 상위 5종목 (설정: `portfolio.top_n`) |
| **동점 처리** | confidence → score → target_return 순으로 정렬 |
""")
