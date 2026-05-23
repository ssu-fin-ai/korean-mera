"""2026년 월별/주별 백테스트 대시보드"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import streamlit as st

from config import ROOT
from db.supabase_store import BacktestStore

BACKTEST_DIR = ROOT / "reports" / "backtest_2026"

_store = BacktestStore()

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


# ── 데이터 로더 (DB 우선, 로컬 fallback) ────────────────────────

def _to_dash(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 and "-" not in d else d

def _from_dash(d: str) -> str:
    return d.replace("-", "") if d else d

@st.cache_data(ttl=300)
def load_picks(port_date, expert):
    rows = _store.get_picks(_to_dash(port_date), expert)
    if rows:
        return rows
    fname = f"{expert}_picks.json" if expert != "combined" else "combined_portfolio.json"
    f = BACKTEST_DIR / port_date / fname
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

@st.cache_data(ttl=300)
def load_combined(port_date):
    rows = _store.get_picks(_to_dash(port_date), "combined")
    if rows:
        return rows
    f = BACKTEST_DIR / port_date / "combined_portfolio.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else []

@st.cache_data(ttl=300)
def load_eval(port_date, eval_date):
    result = _store.get_eval(_to_dash(port_date), _to_dash(eval_date))
    if result:
        return result
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
    ["📋 월별 vs 주별 비교", "📊 분석 주기별 리포트", "📖 전문가 분석 방법론"],
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
        if not df_sum.empty and "expert_key" in df_sum.columns:
            pivot = (
                df_sum[df_sum["expert_key"].isin(EXPERTS)]
                .pivot(index="날짜", columns="전문가", values="평균수익")
                .reindex([_date_label(p, freq_key) for p, _ in sched])
            )
            st.bar_chart(pivot, height=300)
        else:
            st.info("평가 데이터가 없습니다. 백테스트를 먼저 실행하세요.")

        st.divider()

        # 기간별 상세 테이블
        st.subheader("기간별 상세 성과")
        if not df_sum.empty:
            disp = df_sum.copy()
            disp["평균수익"] = disp["평균수익"].apply(lambda x: _fmt(x))
            disp["적중률"] = disp["적중률"].apply(lambda x: f"{x:.0%}")
            st.dataframe(
                disp[["날짜","전문가","종목수","평균수익","적중률"]].style.map(_cr, subset=["평균수익"]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("평가 데이터가 없습니다.")

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

elif section == "📋 월별 vs 주별 비교":
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
                if v > 0:
                    style = "background-color:#1a4d2e; color:#6ee89a; font-weight:bold"
                else:
                    style = "background-color:#4d1a1a; color:#f08080; font-weight:bold"
                return [style]*len(row)
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

    # 최종 통합 포트폴리오 카드
    comb_m = calc_metrics(get_expert_rets("combined", MONTHLY_SCHEDULE), 20)
    comb_w = calc_metrics(get_expert_rets("combined", WEEKLY_SCHEDULE), 5)
    comb_winner = "🔵 월별" if comb_m["cum"] >= comb_w["cum"] else "🟠 주별"
    comb_diff = comb_w["cum"] - comb_m["cum"]
    st.divider()
    st.markdown("#### 🎯 최종 통합 포트폴리오 (5개 전문가 통합 TOP-20)")
    ca, cb, cc, cd, ce = st.columns(5)
    ca.metric("🔵 월별 누적수익", _fmt(comb_m["cum"]),
              delta=f"ARR {_fmt(comb_m['arr'])}", delta_color="normal" if comb_m["arr"] >= 0 else "inverse")
    cb.metric("🔵 월별 MDD", f"{comb_m['mdd']:.1%}")
    cc.metric("🟠 주별 누적수익", _fmt(comb_w["cum"]),
              delta=f"ARR {_fmt(comb_w['arr'])}", delta_color="normal" if comb_w["arr"] >= 0 else "inverse")
    cd.metric("🟠 주별 MDD", f"{comb_w['mdd']:.1%}")
    ce.metric("통합 우위", comb_winner, delta=f"누적수익 차이 {comb_diff:+.1%}",
              delta_color="normal" if comb_diff >= 0 else "inverse")

    # 요약 테이블 (통합 포함)
    comb_row = {
        "전문가": "🎯 통합",
        "🔵 월별 누적수익": comb_m["cum"],
        "🔵 월별 ARR": comb_m["arr"],
        "🔵 월별 MDD": comb_m["mdd"],
        "🟠 주별 누적수익": comb_w["cum"],
        "🟠 주별 ARR": comb_w["arr"],
        "🟠 주별 MDD": comb_w["mdd"],
        "우위": comb_winner,
    }
    disp_sum = pd.DataFrame(sum_rows + [comb_row]).copy()
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
        f"전체 우위: **{overall_pre}** ({m_wins_pre}승 vs {w_wins_pre}승)　│　"
        f"**통합 포트 우위**: {comb_winner} (월 {comb_m['cum']:+.1%} vs 주 {comb_w['cum']:+.1%})"
    )

    st.divider()
    st.markdown("### 전문가별 상세 분석")

    total_wins = {"월별": 0, "주별": 0}

    for expert in ["combined"] + EXPERTS:
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

# ════════════════════════════════════════════════════════════════
# 섹션 3: 전문가 분석 방법론
# ════════════════════════════════════════════════════════════════

elif section == "📖 전문가 분석 방법론":
    st.subheader("📖 전문가 분석 방법론")
    st.caption("MERA 에이전트가 종목을 선정하는 전체 프로세스와 각 전문가의 분석 기준을 설명합니다.")

    # ── 전체 파이프라인 ───────────────────────────────────────────
    st.markdown("### 🔄 전체 종목 선정 프로세스")
    st.markdown("""
| 단계 | 이름 | 설명 |
|------|------|------|
| 1 | **스크리너** | KOSPI200 전체 종목 데이터 수집 → 전문가별 휴리스틱 점수로 Top-15 후보 필터링 |
| 2 | **전문가 에이전트** | 각 전문가가 Top-15 후보를 LLM(Claude)으로 심층 분석 → 최대 5종목 선택 |
| 3 | **어그리게이터** | 5개 전문가의 선정 결과 통합 → 신뢰도 가중 평균으로 최종 TOP-20 구성 |
| 4 | **평가** | 보유 기간 후 실제 수익률 측정 → 적중률·ARR·MDD·Sortino 산출 |
""")

    st.info(
        "**LLM 모델**: Claude Sonnet 4.6 (Anthropic API)  |  "
        "**임베딩**: OpenAI text-embedding-3-small (유사 패턴 검색)  |  "
        "**병렬 처리**: 전문가 5명 동시 실행 (LangGraph fan-out)"
    )
    st.divider()

    # ── 스크리너 ─────────────────────────────────────────────────
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

    # ── 전문가별 상세 ─────────────────────────────────────────────
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
                ("유사 패턴", "OpenAI 임베딩로 검색한 과거 유사 패턴의 5일 수익률 참조"),
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
                    '    "signal": "BUY",          // BUY|HOLD|SELL\n'
                    '    "confidence": 0.85,        // 신뢰도 0~1\n'
                    '    "target_return": 0.10,     // 목표수익 (+10%)\n'
                    '    "horizon_days": 10,         // 투자기간(일)\n'
                    '    "score": 8,                // 매력도 1~10\n'
                    '    "reason": "핵심 근거",\n'
                    '    "pros": ["긍정요인1", ...],\n'
                    '    "cons": ["리스크1", ...]\n'
                    '  }]\n'
                    '}',
                    language="json",
                )

    st.divider()

    # ── 사용 지표 설명 ────────────────────────────────────────────
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
    st.dataframe(
        pd.DataFrame(indicator_data),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # ── 최종 선정 기준 ─────────────────────────────────────────────
    st.markdown("### 4️⃣ 최종 포트폴리오 구성 기준")
    st.markdown("""
5개 전문가가 각각 최대 5종목을 선정한 뒤, 어그리게이터가 다음 기준으로 최종 포트폴리오를 구성합니다.

| 기준 | 내용 |
|------|------|
| **신뢰도 임계값** | confidence ≥ 0.60인 BUY 신호만 채택 |
| **중복 가산점** | 여러 전문가가 동일 종목 추천 시 점수 합산 |
| **최종 종목 수** | 신뢰도 가중 점수 상위 20종목 (설정: `portfolio.top_n`) |
| **동점 처리** | confidence → score → target_return 순으로 정렬 |
""")
    st.caption(
        "설정 파일(config/settings.yaml)에서 `portfolio.top_n`(기본 20), "
        "`portfolio.signal_threshold`(기본 0.60)를 조정할 수 있습니다."
    )
