"""MERA 한국주식 AI 에이전트 — 메인 진입점"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

st.set_page_config(
    page_title="MERA 한국주식",
    page_icon="📊",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/2_백테스트_2026.py", title="📊 백테스트 2026"),
    st.Page("pages/포트폴리오_대시보드.py", title="📈 포트폴리오 대시보드"),
])
pg.run()
