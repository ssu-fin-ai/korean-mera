"""MERA 한국주식 포트폴리오 대시보드"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import streamlit as st

from db.supabase_store import EvaluationStore, PortfolioStore

st.set_page_config(
    page_title="MERA 한국주식 포트폴리오",
    page_icon="📈",
    layout="wide",
)

st.title("📈 MERA 한국주식 포트폴리오 대시보드")

@st.cache_data(ttl=300)
def load_dates():
    return sorted(PortfolioStore().list_dates(), reverse=True)

@st.cache_data(ttl=300)
def load_portfolio(date):
    return PortfolioStore().get_by_date(date)

@st.cache_data(ttl=300)
def load_summary(date):
    return EvaluationStore().get_summary_by_date(date)

@st.cache_data(ttl=300)
def load_stock_evals(date):
    return EvaluationStore().get_stock_evals_by_date(date)

@st.cache_data(ttl=300)
def load_all_summaries():
    return EvaluationStore().get_recent_summaries(30)


# ── 사이드바: 날짜 선택 ───────────────────────────────────────
dates = load_dates()
if not dates:
    st.warning("포트폴리오 데이터가 없습니다. `py main.py --mode run` 을 먼저 실행하세요.")
    st.stop()

selected = st.sidebar.selectbox("📅 날짜 선택", dates)
st.sidebar.markdown("---")
st.sidebar.markdown("**데이터 새로고침**")
if st.sidebar.button("🔄 캐시 초기화"):
    st.cache_data.clear()
    st.rerun()

# ── 주간 성과 요약 ────────────────────────────────────────────
summaries = load_all_summaries()
if summaries:
    st.subheader("주간 성과 요약")
    rows = []
    for s in summaries:
        m = s.get("metadata", s)
        rows.append({
            "포트폴리오 날짜": m["portfolio_date"],
            "평가 날짜": m["eval_date"],
            "평균수익률": m["avg_return"],
            "적중률": m["hit_rate"],
            "ARR": m.get("arr", 0.0) or 0.0,
            "MDD": m.get("mdd", 0.0) or 0.0,
            "소르티노": m.get("sortino", 0.0) or 0.0,
            "칼마": m.get("calmar", 0.0) or 0.0,
            "종목수": m["stock_count"],
        })
    perf_df = pd.DataFrame(rows).sort_values("포트폴리오 날짜")

    cols = st.columns(len(perf_df))
    for col, (_, row) in zip(cols, perf_df.iterrows()):
        ret = row["평균수익률"]
        col.metric(
            label=row["포트폴리오 날짜"],
            value=f"{ret:+.1%}",
            delta=f"적중 {row['적중률']:.0%}",
            delta_color="normal" if ret >= 0 else "inverse",
        )

    st.bar_chart(
        perf_df.set_index("포트폴리오 날짜")["평균수익률"],
        color="#00c49f",
        height=200,
    )
    st.divider()

# ── 선택 날짜 포트폴리오 + 평가 ───────────────────────────────
left, right = st.columns([1, 1], gap="large")

# 왼쪽: 포트폴리오
with left:
    st.subheader(f"🗂 포트폴리오 [{selected}]")
    entries = load_portfolio(selected)
    if entries:
        df = pd.DataFrame(entries)
        show_cols = {
            "rank": "순위", "ticker": "티커", "name": "종목명",
            "sector": "섹터", "score": "점수", "confidence": "신뢰도",
            "target_return": "목표수익",
        }
        df_show = df[[c for c in show_cols if c in df.columns]].rename(columns=show_cols)
        if "신뢰도" in df_show:
            df_show["신뢰도"] = df_show["신뢰도"].apply(lambda x: f"{x:.0%}")
        if "점수" in df_show:
            df_show["점수"] = df_show["점수"].apply(lambda x: f"{x:.2f}")
        if "목표수익" in df_show:
            df_show["목표수익"] = df_show["목표수익"].apply(lambda x: f"{x:+.1%}")
        st.dataframe(df_show, use_container_width=True, hide_index=True)
    else:
        st.info("해당 날짜 포트폴리오가 없습니다.")

# 오른쪽: 평가
with right:
    st.subheader(f"📊 평가 결과 [{selected}]")
    summary = load_summary(selected)

    if summary:
        m = summary.get("metadata", summary)
        avg_ret = m["avg_return"]
        arr = m.get("arr", 0.0) or 0.0
        mdd = m.get("mdd", 0.0) or 0.0
        sortino = m.get("sortino", 0.0) or 0.0
        calmar = m.get("calmar", 0.0) or 0.0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("평균수익률", f"{avg_ret:+.1%}", delta_color="normal" if avg_ret >= 0 else "inverse")
        c2.metric("적중률", f"{m['hit_rate']:.0%}")
        c3.metric("평가 종목", f"{m['stock_count']}개")
        c4.metric("평가 기준일", str(m.get("eval_date", "-")))

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("ARR (연환산)", f"{arr:+.1%}", delta_color="normal" if arr >= 0 else "inverse")
        r2.metric("MDD", f"{mdd:.1%}", delta_color="inverse" if mdd < 0 else "normal")
        r3.metric("소르티노", f"{sortino:.2f}")
        r4.metric("칼마", f"{calmar:.2f}")

        summary_text = summary.get("summary_text", summary.get("text", ""))
        if summary_text:
            with st.expander("🤖 LLM 종합 평가", expanded=True):
                st.write(summary_text)

        stock_evals = load_stock_evals(selected)
        if stock_evals:
            df_e = pd.DataFrame(stock_evals)
            df_e = df_e.sort_values("actual_return", ascending=False)

            show_e = {
                "portfolio_date": "포트폴리오", "eval_date": "평가일",
                "ticker": "티커", "name": "종목명",
                "actual_return": "실제수익", "target_return": "목표수익",
                "correct": "적중",
            }
            df_show_e = df_e[[c for c in show_e if c in df_e.columns]].rename(columns=show_e)
            df_show_e["실제수익"] = df_show_e["실제수익"].apply(lambda x: f"{x:+.1%}")
            df_show_e["목표수익"] = df_show_e["목표수익"].apply(lambda x: f"{x:+.1%}")
            df_show_e["적중"] = df_show_e["적중"].apply(lambda x: "✅ 적중" if x else "❌ 오신호")

            def _style_row(row):
                hit = row.get("적중", "") == "✅ 적중"
                bg = "#0d6832" if hit else "#7b1d1d"
                fg = "#ffffff"
                return [f"background-color: {bg}; color: {fg}; font-weight: bold"] * len(row)

            st.dataframe(
                df_show_e.style.apply(_style_row, axis=1),
                use_container_width=True,
                hide_index=True,
            )

            # 오신호 상위 5개 원인
            wrong = df_e[df_e["correct"] == False].head(5)
            if not wrong.empty and "miss_reason" in wrong.columns:
                with st.expander("❌ 주요 오신호 원인"):
                    for _, row in wrong.iterrows():
                        reason = row.get("miss_reason", "")
                        if reason:
                            st.markdown(f"**{row['ticker']} {row['name']}** ({row['actual_return']:+.1%}): {reason}")
    else:
        st.info("아직 평가 결과가 없습니다.\n\n다음날 `run` 실행 시 자동 평가됩니다.")
