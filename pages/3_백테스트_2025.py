"""2025년 월별 백테스트 대시보드 (12개월)"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

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
    "01": "1월", "02": "2월", "03": "3월", "04": "4월",
    "05": "5월", "06": "6월", "07": "7월", "08": "8월",
    "09": "9월", "10": "10월", "11": "11월", "12": "12월",
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
    ["📊 연간 성과 요약", "📋 월별 상세 리포트", "🗂 포트폴리오 조회"],
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
            .pivot(index="월", columns="전문가", values="평균수익")
            .reindex([_month_label(p) for p, _ in MONTHLY_SCHEDULE])
        )
        st.bar_chart(pivot, height=300)
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
