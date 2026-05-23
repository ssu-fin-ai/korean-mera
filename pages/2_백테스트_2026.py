"""2026년 월별/주별 백테스트 대시보드"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from config import ROOT

st.set_page_config(page_title="MERA 2026 백테스트", page_icon="📊", layout="wide")

BACKTEST_DIR = ROOT / "reports" / "backtest_2026"

MONTHLY_SCHEDULE = [
    ("20260102", "20260130"),
    ("20260202", "20260227"),
    ("20260302", "20260331"),
    ("20260401", "20260430"),
    ("20260504", "20260523"),
]


def _gen_weekly():
    schedule = []
    mondays = pd.date_range("2026-01-05", "2026-05-18", freq="W-MON")
    for mon in mondays:
        port = mon.strftime("%Y%m%d")
        ev = mon + pd.tseries.offsets.BDay(5)
        if ev > pd.Timestamp("2026-05-23"):
            ev = pd.Timestamp("2026-05-23")
        schedule.append((port, ev.strftime("%Y%m%d")))
    return schedule


WEEKLY_SCHEDULE = _gen_weekly()
SCHEDULES = {"월별 (5회)": MONTHLY_SCHEDULE, "주별 (20회)": WEEKLY_SCHEDULE}
HOLD_DAYS_MAP = {"월별 (5회)": 20, "주별 (20회)": 5}
EXPERTS = ["growth", "value", "theme", "dividend", "crisis"]
EXPERT_KOR = {
    "growth": "성장주", "value": "가치주", "theme": "테마주",
    "dividend": "배당주", "crisis": "위기종목", "combined": "통합",
}


# ── 유틸 ────────────────────────────────────────────────────────

def _dash(d): return f"{d[:4]}-{d[4:6]}-{d[6:]}"

def _date_label(port_date, skey):
    if skey == "월별 (5회)":
        return {"01":"1월","02":"2월","03":"3월","04":"4월","05":"5월"}.get(port_date[4:6], port_date)
    return f"{port_date[4:6]}/{port_date[6:]}"

def _fmt(v, f="+.1%"):
    return f"{v:{f}}" if v is not None else "-"


# ── 데이터 로더 ─────────────────────────────────────────────────

@st.cache_data
def load_picks(port_date, expert):
    f = BACKTEST_DIR / port_date / f"{expert}_picks.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

@st.cache_data
def load_combined(port_date):
    f = BACKTEST_DIR / port_date / "combined_portfolio.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

@st.cache_data
def load_eval(port_date, eval_date):
    f = BACKTEST_DIR / port_date / f"eval_{eval_date}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


# ── 지표 계산 ────────────────────────────────────────────────────

def calc_metrics(rets: list[float], hold_days: int) -> dict:
    if not rets:
        return {"cum": 0.0, "arr": 0.0, "mdd": 0.0, "sortino": 0.0, "calmar": 0.0, "hit": 0.0}
    r = np.array(rets)
    ann = 252 / hold_days
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

def get_expert_rets(expert, sched):
    return [r for r in
            (load_eval(p,e).get(expert,{}).get("avg_return") for p,e in sched
             if load_eval(p,e).get(expert))
            if r is not None]

def build_summary_df(skey):
    sched = SCHEDULES[skey]
    rows = []
    for port_date, eval_date in sched:
        ev = load_eval(port_date, eval_date)
        for expert in EXPERTS + ["combined"]:
            if expert not in ev:
                continue
            r = ev[expert]
            rows.append({
                "날짜": _date_label(port_date, skey),
                "port_date": port_date,
                "전문가": EXPERT_KOR.get(expert, expert),
                "expert_key": expert,
                "평균수익": r["avg_return"],
                "적중률": r["hit_rate"],
                "종목수": r["stock_count"],
            })
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════
# 페이지 상단: 섹션 분리
# ════════════════════════════════════════════════════════════════

st.title("📊 MERA 2026 백테스트")
st.caption("전문가별 포트폴리오 선정 및 실제 수익률 평가 | 기간: 2026.01 ~ 05")

section = st.radio(
    "섹션 선택",
    ["📊 분석 주기별 리포트", "📋 월별 vs 주별 비교"],
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()


# ════════════════════════════════════════════════════════════════
# 섹션 1: 분석 주기별 리포트
# ════════════════════════════════════════════════════════════════

if section == "📊 분석 주기별 리포트":

    freq_key = st.radio("분석 주기", list(SCHEDULES.keys()), horizontal=True)
    sched = SCHEDULES[freq_key]
    hd = HOLD_DAYS_MAP[freq_key]
    eval_map = dict(sched)
    date_labels = {_date_label(p, freq_key): p for p, _ in sched}

    tab_sum, tab_port, tab_eval = st.tabs(["📈 성과 요약", "🗂 포트폴리오 조회", "🎯 평가 상세"])

    # ── 성과 요약 ──────────────────────────────────────────────
    with tab_sum:
        n = len(sched)
        df_sum = build_summary_df(freq_key)

        # 전문가별 누적/ARR/MDD 카드
        st.subheader(f"전문가별 누적 성과 ({n}회, {freq_key[:2]}별)")
        cols = st.columns(len(EXPERTS))
        for col, expert in zip(cols, EXPERTS):
            rets = get_expert_rets(expert, sched)
            m = calc_metrics(rets, hd)
            col.metric(
                EXPERT_KOR[expert],
                f"누적 {m['cum']:+.1%}",
                delta=f"ARR {m['arr']:+.1%}",
                delta_color="normal" if m["arr"] >= 0 else "inverse",
            )

        st.divider()

        # 리스크 지표 테이블
        st.subheader("전문가별 리스크 지표")
        risk_rows = []
        for expert in EXPERTS + ["combined"]:
            rets = get_expert_rets(expert, sched)
            m = calc_metrics(rets, hd)
            risk_rows.append({
                "전문가": EXPERT_KOR.get(expert, expert),
                f"누적수익({n}회)": _fmt(m["cum"]),
                "ARR (연환산)": _fmt(m["arr"]),
                "MDD": f"{m['mdd']:.1%}",
                "Sortino": f"{m['sortino']:.2f}",
                "Calmar": f"{m['calmar']:.2f}",
                "평균 적중률": f"{m['hit']:.0%}",
            })

        def _cr(val):
            try:
                v = float(str(val).replace("%","").replace("+",""))
                return f"color:{'#2ecc71' if v>0 else '#e74c3c'};font-weight:bold"
            except: return ""

        st.dataframe(
            pd.DataFrame(risk_rows).style.map(_cr, subset=[f"누적수익({n}회)", "ARR (연환산)"]),
            use_container_width=True, hide_index=True,
        )

        st.divider()

        # 기간별 수익률 차트
        st.subheader("기간별 평균수익률")
        pivot = (
            df_sum[df_sum["expert_key"].isin(EXPERTS)]
            .pivot(index="날짜", columns="전문가", values="평균수익")
            .reindex([_date_label(p, freq_key) for p, _ in sched])
        )
        st.bar_chart(pivot, height=300)

        st.divider()

        # 기간별 상세 테이블
        st.subheader("기간별 상세 성과")
        disp = df_sum.copy()
        disp["평균수익"] = disp["평균수익"].apply(lambda x: _fmt(x))
        disp["적중률"] = disp["적중률"].apply(lambda x: f"{x:.0%}")
        st.dataframe(
            disp[["날짜","전문가","종목수","평균수익","적중률"]].style.map(_cr, subset=["평균수익"]),
            use_container_width=True, hide_index=True,
        )

    # ── 포트폴리오 조회 ────────────────────────────────────────
    with tab_port:
        sel_label = st.selectbox("📅 날짜 선택", list(date_labels.keys()))
        sel_port = date_labels[sel_label]
        sel_eval = eval_map[sel_port]
        ev_port = load_eval(sel_port, sel_eval)
        st.caption(f"포트폴리오: {_dash(sel_port)}  →  평가일: {_dash(sel_eval)}")

        expert_tabs = st.tabs([EXPERT_KOR[e] for e in EXPERTS] + ["통합"])

        def _render_port(picks, ev_data):
            if not picks:
                st.info("선정 종목 없음")
                return
            buy = [p for p in picks if p.get("signal") == "BUY"] or picks
            ev_map = {s["ticker"]: s for s in (ev_data.get("stocks") if ev_data else [])}

            if ev_data:
                m1, m2, m3 = st.columns(3)
                ret = ev_data.get("avg_return", 0)
                m1.metric("평균수익률", _fmt(ret), delta_color="normal" if ret >= 0 else "inverse")
                m2.metric("적중률", f"{ev_data.get('hit_rate',0):.0%}")
                m3.metric("종목수", f"{ev_data.get('stock_count', len(buy))}개")

            rows = []
            for i, p in enumerate(buy, 1):
                t = p.get("ticker","")
                ev = ev_map.get(t, {})
                rows.append({
                    "#": i, "티커": t, "종목명": p.get("name",""),
                    "신뢰도": f"{p.get('confidence',0):.0%}",
                    "목표수익": _fmt(p.get("target_return") or p.get("avg_target_return") or 0),
                    "실제수익": _fmt(ev.get("actual_return")) if ev.get("actual_return") is not None else "-",
                    "적중": ("✅" if ev["hit"] else "❌") if "hit" in ev else "-",
                })
            df_p = pd.DataFrame(rows)

            def _sc(val):
                if val == "-": return ""
                try:
                    v = float(str(val).replace("%","").replace("+",""))
                    return f"color:{'#2ecc71' if v>0 else '#e74c3c'};font-weight:bold"
                except: return ""

            st.dataframe(df_p.style.map(_sc, subset=["실제수익"]),
                         use_container_width=True, hide_index=True)

            with st.expander("선정이유 / 긍정 / 리스크"):
                for p in buy:
                    st.markdown(f"**{p.get('ticker','')} {p.get('name','')}**")
                    if p.get("reason"): st.caption(f"📌 {p['reason']}")
                    c1, c2 = st.columns(2)
                    pros = " / ".join((p.get("pros") or [])[:2])
                    cons = " / ".join((p.get("cons") or [])[:2])
                    if pros: c1.markdown(f"✅ {pros}")
                    if cons: c2.markdown(f"⚠️ {cons}")
                    st.markdown("---")

        for i, exp in enumerate(EXPERTS):
            with expert_tabs[i]:
                _render_port(load_picks(sel_port, exp), ev_port.get(exp))
        with expert_tabs[-1]:
            _render_port(load_combined(sel_port), ev_port.get("combined"))

    # ── 평가 상세 ──────────────────────────────────────────────
    with tab_eval:
        sel_el = st.selectbox("📅 날짜 선택", list(date_labels.keys()), key="ev_sel")
        sel_ep = date_labels[sel_el]
        sel_ed = eval_map[sel_ep]
        ev = load_eval(sel_ep, sel_ed)
        if not ev:
            st.info("평가 데이터 없음")
        else:
            st.caption(f"포트폴리오: {_dash(sel_ep)}  →  평가일: {_dash(sel_ed)}")

            mcols = st.columns(len(EXPERTS))
            for col, exp in zip(mcols, EXPERTS):
                if exp not in ev:
                    col.metric(EXPERT_KOR[exp], "-"); continue
                r = ev[exp]; ret = r["avg_return"]
                col.metric(EXPERT_KOR[exp], _fmt(ret),
                           delta=f"적중 {r['hit_rate']:.0%}/{r['stock_count']}종목",
                           delta_color="normal" if ret >= 0 else "inverse")
            st.divider()

            # 전체 기간 누적 리스크
            st.subheader(f"전체 기간 누적 리스크 ({n}회)")
            rsk_rows = []
            for exp in EXPERTS + ["combined"]:
                rets = get_expert_rets(exp, sched)
                m = calc_metrics(rets, hd)
                rsk_rows.append({
                    "전문가": EXPERT_KOR.get(exp, exp),
                    f"누적수익({n}회)": _fmt(m["cum"]),
                    "ARR": _fmt(m["arr"]),
                    "MDD": f"{m['mdd']:.1%}",
                    "Sortino": f"{m['sortino']:.2f}",
                    "Calmar": f"{m['calmar']:.2f}",
                })
            st.dataframe(pd.DataFrame(rsk_rows).style.map(_cr, subset=[f"누적수익({n}회)","ARR"]),
                         use_container_width=True, hide_index=True)
            st.divider()

            def _sr(val):
                try:
                    v = float(str(val).replace("%","").replace("+",""))
                    return f"color:{'#2ecc71' if v>0 else '#e74c3c'};font-weight:bold"
                except: return ""

            st.subheader("전문가별 종목 실적")
            for exp in EXPERTS + ["combined"]:
                if exp not in ev: continue
                r = ev[exp]; ret = r["avg_return"]
                label = EXPERT_KOR.get(exp, exp)
                st.markdown(
                    f"#### {'📈' if ret>=0 else '📉'} {label}"
                    f"　평균수익: **{_fmt(ret)}**"
                    f"　적중률: **{r['hit_rate']:.0%}**"
                    f"　({r['stock_count']}종목)"
                )
                stocks = sorted(r["stocks"], key=lambda x: x["actual_return"], reverse=True)
                rows = [{"티커":s["ticker"],"종목명":s["name"],
                         "목표수익":s["target_return"],"실제수익":s["actual_return"],
                         "적중":"✅" if s["hit"] else "❌","선정이유":s.get("reason","")}
                        for s in stocks]
                df_s = pd.DataFrame(rows)
                cc, ct = st.columns([1,2], gap="medium")
                with cc:
                    st.bar_chart(df_s.set_index("종목명")[["목표수익","실제수익"]], height=200)
                with ct:
                    d2 = df_s.copy()
                    d2["목표수익"] = d2["목표수익"].apply(lambda x: _fmt(x))
                    d2["실제수익"] = d2["실제수익"].apply(lambda x: _fmt(x))
                    st.dataframe(d2.style.map(_sr, subset=["실제수익"]),
                                 use_container_width=True, hide_index=True)
                st.divider()


# ════════════════════════════════════════════════════════════════
# 섹션 2: 월별 vs 주별 비교
# ════════════════════════════════════════════════════════════════

else:
    st.subheader("📋 월별 vs 주별 상세 비교")
    st.caption(
        "🔵 월별: 5회 (보유 ~20일)  |  🟠 주별: 20회 (보유 5일)  |  "
        "**우위 판정**: ① 5개월 누적수익 높은 쪽 → 동점 시 ② MDD 낙폭 적은 쪽  |  "
        "**리스크 점수**: ARR(연환산) / MDD(최대낙폭) / Sortino(하방변동성 대비 수익) / Calmar(ARR÷|MDD|)"
    )

    def _render_side(expert, sched, hd, label_skey):
        rets_by_period = []
        period_labels = []
        stock_map: dict[str, dict] = {}
        for port_date, eval_date in sched:
            ev = load_eval(port_date, eval_date)
            if expert not in ev: continue
            r = ev[expert]
            rets_by_period.append(r["avg_return"])
            period_labels.append(_date_label(port_date, label_skey))
            for s in r.get("stocks", []):
                t = s["ticker"]
                if t not in stock_map:
                    stock_map[t] = {"name":s.get("name",""), "returns":[], "hits":[]}
                stock_map[t]["returns"].append(s["actual_return"])
                stock_map[t]["hits"].append(s["hit"])

        m = calc_metrics(rets_by_period, hd)

        # 핵심 지표
        a, b, c, d = st.columns(4)
        a.metric("누적수익", _fmt(m["cum"]), delta_color="normal" if m["cum"]>=0 else "inverse")
        b.metric("ARR (연환산)", _fmt(m["arr"]), delta_color="normal" if m["arr"]>=0 else "inverse")
        c.metric("MDD", f"{m['mdd']:.1%}", delta_color="inverse" if m["mdd"]<0 else "normal")
        d.metric("Sortino / Calmar", f"{m['sortino']:.2f} / {m['calmar']:.2f}")
        e_, f_ = st.columns(2)
        e_.metric("평균 적중률", f"{m['hit']:.0%}")
        f_.metric("총 회차", f"{len(rets_by_period)}회")

        # 기간별 추이
        if rets_by_period:
            st.markdown("**기간별 수익률 추이**")
            st.bar_chart(pd.DataFrame({"수익률": rets_by_period}, index=period_labels), height=160)

        # 종목 테이블
        if not stock_map:
            st.info("종목 없음"); return m, pd.DataFrame()

        df_st = pd.DataFrame([
            {"티커":t, "종목명":v["name"],
             "선정횟수":len(v["returns"]),
             "평균수익": sum(v["returns"])/len(v["returns"]),
             "최고": max(v["returns"]),
             "최저": min(v["returns"]),
             "적중률": sum(v["hits"])/len(v["hits"])}
            for t,v in stock_map.items()
        ]).sort_values("평균수익", ascending=False)

        st.markdown("**종목별 성과**")
        disp = df_st.copy()
        for col in ["평균수익","최고","최저"]:
            disp[col] = disp[col].apply(lambda x: _fmt(x))
        disp["적중률"] = disp["적중률"].apply(lambda x: f"{x:.0%}")

        def _row_bg(row):
            try:
                v = float(str(row["평균수익"]).replace("%","").replace("+",""))
                bg = "#0d3320" if v>0 else "#3a0d0d"
                return [f"background-color:{bg}"]*len(row)
            except: return [""]*len(row)

        st.dataframe(disp.style.apply(_row_bg, axis=1),
                     use_container_width=True, hide_index=True)

        top3 = df_st.head(3); bot3 = df_st.tail(3)
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("🏆 **상위 3종목**")
            for _, row in top3.iterrows():
                st.markdown(f"- **{row['종목명']}** {_fmt(row['평균수익'])} ({row['선정횟수']}회)")
        with t2:
            st.markdown("⚠️ **하위 3종목**")
            for _, row in bot3.iterrows():
                st.markdown(f"- **{row['종목명']}** {_fmt(row['평균수익'])} ({row['선정횟수']}회)")

        return m, df_st

    # ── 상단 요약 ────────────────────────────────────────────────
    st.markdown("### 📌 전문가별 누적수익 한눈에 보기")

    sum_rows = []
    best_m_exp, best_w_exp = None, None
    best_m_cum, best_w_cum = -999, -999
    for exp in EXPERTS:
        m_r = calc_metrics(get_expert_rets(exp, MONTHLY_SCHEDULE), 20)
        w_r = calc_metrics(get_expert_rets(exp, WEEKLY_SCHEDULE), 5)
        label_ = EXPERT_KOR[exp]
        winner_ = "🔵 월별" if m_r["cum"] >= w_r["cum"] else "🟠 주별"
        sum_rows.append({
            "전문가": label_,
            "🔵 월별 누적수익": m_r["cum"],
            "🔵 월별 ARR": m_r["arr"],
            "🔵 월별 MDD": m_r["mdd"],
            "🟠 주별 누적수익": w_r["cum"],
            "🟠 주별 ARR": w_r["arr"],
            "🟠 주별 MDD": w_r["mdd"],
            "우위": winner_,
        })
        if m_r["cum"] > best_m_cum:
            best_m_cum, best_m_exp = m_r["cum"], label_
        if w_r["cum"] > best_w_cum:
            best_w_cum, best_w_exp = w_r["cum"], label_

    # 메트릭 카드 (전문가 × 월별/주별)
    expert_cols = st.columns(len(EXPERTS))
    for col, row in zip(expert_cols, sum_rows):
        diff = row["🟠 주별 누적수익"] - row["🔵 월별 누적수익"]
        col.metric(
            row["전문가"],
            f"월 {row['🔵 월별 누적수익']:+.1%}",
            delta=f"주 {row['🟠 주별 누적수익']:+.1%}  (차이 {diff:+.1%})",
            delta_color="normal" if diff >= 0 else "inverse",
        )

    st.divider()

    # 요약 테이블
    disp_sum = pd.DataFrame(sum_rows).copy()
    for c in ["🔵 월별 누적수익","🔵 월별 ARR","🟠 주별 누적수익","🟠 주별 ARR"]:
        disp_sum[c] = disp_sum[c].apply(lambda x: _fmt(x))
    for c in ["🔵 월별 MDD","🟠 주별 MDD"]:
        disp_sum[c] = disp_sum[c].apply(lambda x: f"{x:.1%}")
    st.dataframe(disp_sum, use_container_width=True, hide_index=True)

    # 분류 및 하이라이트
    m_wins_pre = sum(1 for r in sum_rows if "월별" in r["우위"])
    w_wins_pre = len(sum_rows) - m_wins_pre
    overall_pre = "🔵 월별" if m_wins_pre >= w_wins_pre else "🟠 주별"

    monthly_experts = [r["전문가"] for r in sum_rows if "월별" in r["우위"]]
    weekly_experts  = [r["전문가"] for r in sum_rows if "주별"  in r["우위"]]

    cl, cr = st.columns(2)
    with cl:
        st.info(
            f"**🔵 월별 리밸런싱 적합 전문가** ({len(monthly_experts)}개)\n\n"
            + "\n".join(
                f"- **{r['전문가']}**　월 {r['🔵 월별 누적수익']:+.1%} vs 주 {r['🟠 주별 누적수익']:+.1%}"
                for r in sum_rows if "월별" in r["우위"]
            )
            + "\n\n*추세가 월 단위로 형성되어 장기 보유 시 더 유리*"
        )
    with cr:
        st.info(
            f"**🟠 주별 리밸런싱 적합 전문가** ({len(weekly_experts)}개)\n\n"
            + "\n".join(
                f"- **{r['전문가']}**　주 {r['🟠 주별 누적수익']:+.1%} vs 월 {r['🔵 월별 누적수익']:+.1%}"
                for r in sum_rows if "주별" in r["우위"]
            )
            + "\n\n*단기 이슈·모멘텀에 민감하여 빠른 회전이 더 유리*"
        )

    st.success(
        f"**월별 최고**: {best_m_exp} ({best_m_cum:+.1%})　"
        f"**주별 최고**: {best_w_exp} ({best_w_cum:+.1%})　│　"
        f"전체 우위: **{overall_pre}** ({m_wins_pre}승 vs {w_wins_pre}승)"
    )

    st.divider()
    st.markdown("### 전문가별 상세 분석")

    total_wins = {"월별": 0, "주별": 0}

    for expert in EXPERTS + ["combined"]:
        label = EXPERT_KOR.get(expert, expert)
        st.markdown("---")

        # 헤더: 미리 계산
        m_rets = get_expert_rets(expert, MONTHLY_SCHEDULE)
        w_rets = get_expert_rets(expert, WEEKLY_SCHEDULE)
        mm = calc_metrics(m_rets, 20)
        wm = calc_metrics(w_rets, 5)
        # 누적수익 기준 우위 판정 (같으면 MDD로 보조 판단)
        if mm["cum"] != wm["cum"]:
            winner = "🔵 월별" if mm["cum"] >= wm["cum"] else "🟠 주별"
            basis = "누적수익"
        else:
            winner = "🔵 월별" if mm["mdd"] >= wm["mdd"] else "🟠 주별"
            basis = "MDD"
        if expert != "combined":
            total_wins["월별" if "월별" in winner else "주별"] += 1

        st.markdown(
            f"## {label} 전문가\n"
            f"🔵 월별 누적: **{_fmt(mm['cum'])}** (ARR {_fmt(mm['arr'])})　"
            f"🟠 주별 누적: **{_fmt(wm['cum'])}** (ARR {_fmt(wm['arr'])})　"
            f"→ **{winner} 우위** *(기준: {basis})*"
        )

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("### 🔵 월별 (5회, 보유 ~20일)")
            mm_full, m_df = _render_side(expert, MONTHLY_SCHEDULE, 20, "월별 (5회)")
        with right:
            st.markdown("### 🟠 주별 (20회, 보유 5일)")
            wm_full, w_df = _render_side(expert, WEEKLY_SCHEDULE, 5, "주별 (20회)")

        # 공통 종목 교차 분석
        if not m_df.empty and not w_df.empty:
            common = set(m_df["티커"]) & set(w_df["티커"])
            if common:
                st.markdown("#### 🔄 공통 선정 종목 비교")
                mi = m_df.set_index("티커"); wi = w_df.set_index("티커")
                cross = []
                for t in sorted(common):
                    mr_ = mi.loc[t]; wr_ = wi.loc[t]
                    diff = mr_["평균수익"] - wr_["평균수익"]
                    cross.append({
                        "티커": t, "종목명": mr_["종목명"],
                        "월별 평균수익": _fmt(mr_["평균수익"]),
                        "월별 선정횟수": int(mr_["선정횟수"]),
                        "주별 평균수익": _fmt(wr_["평균수익"]),
                        "주별 선정횟수": int(wr_["선정횟수"]),
                        "차이(월−주)": _fmt(diff),
                        "유리": "🔵 월별" if diff >= 0 else "🟠 주별",
                    })
                st.dataframe(pd.DataFrame(cross), use_container_width=True, hide_index=True)

    # ── 최종 판정 ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🏁 최종 종합 판정")
    overall = "🔵 월별" if total_wins["월별"] >= total_wins["주별"] else "🟠 주별"

    j1, j2, j3 = st.columns(3)
    j1.metric("🔵 월별 우위 전문가", f"{total_wins['월별']} / 5")
    j2.metric("🟠 주별 우위 전문가", f"{total_wins['주별']} / 5")
    j3.metric("최종 승자", overall)

    arr_data = {"전문가": [], "월별 ARR": [], "주별 ARR": [],
                "월별 누적수익": [], "주별 누적수익": []}
    for exp in EXPERTS:
        arr_data["전문가"].append(EXPERT_KOR[exp])
        for key, sched, hd_, arr_col, cum_col in [
            ("월별", MONTHLY_SCHEDULE, 20, "월별 ARR", "월별 누적수익"),
            ("주별", WEEKLY_SCHEDULE, 5, "주별 ARR", "주별 누적수익"),
        ]:
            rets = get_expert_rets(exp, sched)
            m_ = calc_metrics(rets, hd_)
            arr_data[arr_col].append(m_["arr"])
            arr_data[cum_col].append(m_["cum"])

    final_df = pd.DataFrame(arr_data).set_index("전문가")
    c_arr, c_cum = st.columns(2)
    with c_arr:
        st.markdown("#### ARR 비교")
        st.bar_chart(final_df[["월별 ARR","주별 ARR"]], height=260)
    with c_cum:
        st.markdown("#### 5개월 누적수익 비교")
        st.bar_chart(final_df[["월별 누적수익","주별 누적수익"]], height=260)

    st.info(
        f"**결론: {overall} 리밸런싱 우위**\n\n"
        "| | 🔵 월별 | 🟠 주별 |\n"
        "|--|--|--|\n"
        "| 거래 비용 | 낮음 (5회) | 높음 (20회) |\n"
        "| 추세 반영 | 월 단위 큰 흐름 | 주 단위 빠른 대응 |\n"
        "| 손실 회복 | 느림 | 빠름 |\n"
        "| 변동성 | 낮음 | 높음 |\n\n"
        "*점수 기준: ARR − |MDD| + Sortino × 0.1*"
    )
