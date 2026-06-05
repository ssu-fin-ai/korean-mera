"""Korean-MERA 아키텍처 흐름도 3종 생성 (PNG).

출력:
  docs/report/정리/01_DB구축_흐름도.png
  docs/report/정리/02_종목선정_흐름도.png
  docs/report/정리/03_종목평가_흐름도.png
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows Malgun Gothic)
_kr_font = next(
    (f.fname for f in fm.fontManager.ttflist if f.name == "Malgun Gothic"), None
)
if _kr_font:
    plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = Path(__file__).parent / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 색상 팔레트 (숭실대 공식) ─────────────────────────────────────────────────
C_BLUE      = "#006494"
C_MID_BLUE  = "#009DB1"
C_TEAL      = "#6EC4C0"
C_DARK      = "#1C2B3A"
C_LIGHT     = "#F0F7FA"
C_GRAY      = "#88A0AE"
C_WHITE     = "#FFFFFF"
C_GREEN     = "#27AE60"
C_ORANGE    = "#E67E22"
C_RED       = "#C0392B"


def _box(ax, x, y, w, h, label, bg, fg=C_WHITE, fontsize=10, bold=False,
         radius=0.03, sub=None, sub_fg=None):
    """둥근 사각형 박스 + 텍스트 (+ 선택적 서브텍스트)."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0, facecolor=bg, zorder=3,
    )
    ax.add_patch(box)
    va = "center" if sub is None else "top"
    ty = y + h * 0.12 if sub else y
    ax.text(
        x, ty, label,
        ha="center", va=va, fontsize=fontsize,
        color=fg, fontweight="bold" if bold else "normal",
        zorder=4, wrap=False,
    )
    if sub:
        ax.text(
            x, y - h * 0.18, sub,
            ha="center", va="center", fontsize=fontsize - 1.5,
            color=sub_fg or C_GRAY, zorder=4,
        )


def _arrow(ax, x1, y1, x2, y2, color=C_GRAY, lw=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color,
                        lw=lw, mutation_scale=14),
        zorder=2,
    )


def _hline(ax, y, x0, x1, color=C_GRAY, lw=1.2, ls="--"):
    ax.plot([x0, x1], [y, y], color=color, lw=lw, ls=ls, zorder=2)


def _label(ax, x, y, text, color=C_GRAY, fontsize=8.5, ha="left"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, zorder=5)


# ══════════════════════════════════════════════════════════════════════════════
# 01  DB 구축 흐름도  (build_history_db)
# ══════════════════════════════════════════════════════════════════════════════

def make_db_build():
    fig, ax = plt.subplots(figsize=(9, 14))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    # 제목
    ax.text(4.5, 13.5, "01  DB 구축 흐름도", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C_BLUE)
    ax.text(4.5, 13.1, "build_history_db()  ·  scheduler/pipeline.py",
            ha="center", va="center", fontsize=9, color=C_GRAY)
    ax.plot([0.5, 8.5], [12.85, 12.85], color=C_TEAL, lw=2)

    cx = 4.5
    bw, bh = 6.0, 0.62

    steps = [
        (12.3, "KOSPI200 종목 수집",          "get_universe()  →  pykrx 인덱스 코드 1028",     C_BLUE,     C_GRAY),
        (11.3, "OHLCV 수집 (일봉)",            "pykrx  →  FDR fallback  |  캐시: {ticker}_{start}_{end}.parquet", C_BLUE, C_GRAY),
        (10.3, "기술 지표 계산",               "feature_engineer.compute()\nRSI · MACD · BB · ADX · MFI · 수익률 · 거래량",  C_MID_BLUE, C_GRAY),
        ( 9.3, "상대강도 계산",                "add_relative_strength()  ·  KOSPI 대비 rel_strength_5d/20d · beta_20d",        C_MID_BLUE, C_GRAY),
        ( 8.3, "스냅샷 벡터 추출",             "get_snapshot_vector()  ·  날짜별 12개 스칼라 + returns_series(20일)",            C_MID_BLUE, C_GRAY),
        ( 7.3, "레이블 산출",                  "label_5d = pct_change(5).shift(-5)\nlabel_10d · label_20d 동일 방식",           C_ORANGE,   C_GRAY),
        ( 6.3, "자연어 텍스트 생성",           "generate_pattern_text()  ·  data/text_generator.py",                           C_TEAL,     C_DARK),
        ( 5.3, "OpenAI 임베딩",                "text-embedding-3-small  →  1,536차원 벡터",                                    C_GREEN,    C_DARK),
        ( 4.3, "ChromaDB 저장",                "pattern_store.upsert_batch()\nID: {ticker}_{date}  |  컬렉션: stock_patterns",  C_BLUE,     C_GRAY),
    ]

    for y, title, sub, bg, sfg in steps:
        _box(ax, cx, y, bw, bh, title, bg, fontsize=11, bold=True,
             sub=sub, sub_fg=sfg)
        if y > 4.3:
            _arrow(ax, cx, y - bh / 2, cx, y - bh / 2 - 0.36)

    # 메타데이터 주석 박스
    meta_items = [
        "ticker  ·  name  ·  sector  ·  market",
        "date  ·  label_5d  ·  label_10d  ·  label_20d",
    ]
    mx, my, mw, mh = 7.4, 4.3, 1.6, 0.75
    _box(ax, mx, my, mw, mh, "메타데이터", C_LIGHT, fg=C_DARK, fontsize=9)
    for i, item in enumerate(meta_items):
        ax.text(mx, my + 0.08 - i * 0.22, item,
                ha="center", va="center", fontsize=7.5, color=C_DARK, zorder=5)
    ax.annotate("", xy=(cx + bw / 2, 4.3), xytext=(mx - mw / 2, my),
                arrowprops=dict(arrowstyle="-|>", color=C_TEAL, lw=1.2,
                                mutation_scale=11), zorder=2)

    # 규모 주석
    _box(ax, cx, 3.0, 6.0, 0.7,
         "전체 규모:  ~200종목 × 5년 ≈ 약 250,000건 적재",
         C_LIGHT, fg=C_DARK, fontsize=10)
    _arrow(ax, cx, 4.3 - bh / 2, cx, 3.0 + 0.35)

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "01_DB구축_흐름도.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    print(f"저장: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 02  종목선정 흐름도  (screener_node)
# ══════════════════════════════════════════════════════════════════════════════

def make_screener():
    fig, ax = plt.subplots(figsize=(12, 13))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 13)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    ax.text(6.0, 12.5, "02  종목선정 흐름도", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C_BLUE)
    ax.text(6.0, 12.1, "screener_node()  ·  agents/screener.py",
            ha="center", va="center", fontsize=9, color=C_GRAY)
    ax.plot([0.5, 11.5], [11.88, 11.88], color=C_TEAL, lw=2)

    # ── 상단: 입력 ──
    _box(ax, 6.0, 11.4, 5.5, 0.62,
         "KOSPI200 전체 종목", C_BLUE, fontsize=12, bold=True,
         sub="get_universe()  |  설정: kospi200=true, kosdaq150=false",
         sub_fg=C_GRAY)

    _arrow(ax, 6.0, 11.09, 6.0, 10.62)

    # ── 수집 블록 ──
    _box(ax, 6.0, 10.3, 11.0, 0.62,
         "_collect_ticker()  [ThreadPoolExecutor  workers=3]",
         C_MID_BLUE, fontsize=11, bold=True,
         sub="종목별 병렬 수집  |  데이터 조회 + 지표 계산 + RAG 검색",
         sub_fg=C_GRAY)

    # 수집 항목 5개
    data_items = [
        (1.1,  9.1, 1.8, "OHLCV\n기술지표",     "feature_engineer\n.compute()",  C_BLUE),
        (3.2,  9.1, 1.8, "재무지표",             "pykrx · DART\n네이버 fallback", C_BLUE),
        (5.3,  9.1, 1.8, "공매도",               "short_{ticker}\n_{date}.json",  C_BLUE),
        (7.4,  9.1, 1.8, "DART 공시",            "B·D·E\n최근 30일",              C_BLUE),
        (9.5,  9.1, 1.8, "RAG 유사패턴",         "ChromaDB\nTop-5 검색",          C_GREEN),
    ]
    for dx, dy, dw, title, sub, bg in data_items:
        _box(ax, dx, dy, dw, 0.88, title, bg, fontsize=10, bold=True,
             sub=sub, sub_fg=C_GRAY)
        _arrow(ax, dx, 9.99, dx, dy + 0.44, color=C_MID_BLUE)

    # 재무지표 세부 항목
    fin_details = [
        "PER · PBR · EPS · BPS",
        "DIV · DPS  (pykrx)",
        "매출 · 영업이익 · 순이익",
        "ROE · ROA · 부채비율",
        "Graham Number · FCF",
    ]
    fx, fy = 3.2, 7.9
    _box(ax, fx, fy, 2.2, 1.2, "", C_LIGHT, fontsize=8.5)
    for i, item in enumerate(fin_details):
        ax.text(fx, fy + 0.38 - i * 0.22, item,
                ha="center", va="center", fontsize=7.5, color=C_DARK, zorder=5)
    _arrow(ax, fx, 8.66, fx, fy + 0.6, color=C_GRAY)

    # 섹터비교
    _box(ax, 3.2, 7.05, 2.2, 0.52,
         "섹터 대비 PER/PBR", C_LIGHT, fg=C_DARK, fontsize=8.5,
         sub="get_sector_avg_fundamental()", sub_fg=C_GRAY)
    _arrow(ax, fx, 7.3, fx, 7.05 + 0.26, color=C_GRAY)

    # ── 스코어링 구분선 ──
    ax.plot([0.4, 11.6], [6.55, 6.55], color=C_TEAL, lw=1.2, ls="--")
    ax.text(6.0, 6.42, "전문가별 휴리스틱 스코어링",
            ha="center", va="center", fontsize=9, color=C_TEAL, fontstyle="italic")

    # 5개 전문가 스코어 박스
    experts = [
        (1.1,  5.6, "growth\n성장주",
         "volume_ratio\nMACD·RSI\nret_5d\nop/rev_yoy", C_BLUE),
        (3.2,  5.6, "value\n가치주",
         "PBR·PER\nper/pbr_vs\nROE·debt", C_BLUE),
        (5.3,  5.6, "theme\n테마주",
         "volume_ratio\nret_5d·RSI\nshort_ratio", C_BLUE),
        (7.4,  5.6, "dividend\n배당주",
         "DIV·payout\nFCF·beta\ndebt_ratio", C_BLUE),
        (9.5,  5.6, "crisis\n위기종목",
         "ret_5d·RSI\nbb_pct·debt\ncurrent_ratio", C_RED),
    ]
    for ex, ey, title, sub, bg in experts:
        _box(ax, ex, ey, 1.9, 1.5, title, bg, fontsize=10, bold=True,
             sub=sub, sub_fg=C_GRAY)
        _arrow(ax, ex, 6.3, ex, ey + 0.75, color=C_MID_BLUE)

    # ── Top-N 선정 ──
    _arrow(ax, 6.0, 4.85, 6.0, 4.42)
    _box(ax, 6.0, 4.1, 10.5, 0.62,
         "전문가별 Top-15 후보 선정",
         C_MID_BLUE, fontsize=11, bold=True,
         sub="screener.TOP_N = 15  |  각 전문가가 독립적으로 상위 15종목 선별",
         sub_fg=C_GRAY)

    # ── 출력 ──
    _arrow(ax, 6.0, 3.79, 6.0, 3.32)
    _box(ax, 6.0, 3.0, 10.5, 0.62,
         "state: {growth/value/theme/dividend/crisis}_candidates",
         C_DARK, fontsize=10, bold=False,
         sub="LangGraph state에 전달  →  gate_node 입력",
         sub_fg=C_TEAL)

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "02_종목선정_흐름도.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    print(f"저장: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 03  종목평가 흐름도  (gate → experts → aggregator)
# ══════════════════════════════════════════════════════════════════════════════

def make_evaluation():
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis("off")
    fig.patch.set_facecolor(C_WHITE)

    ax.text(6.0, 13.5, "03  종목평가 흐름도", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C_BLUE)
    ax.text(6.0, 13.1, "gate_node → experts → aggregator_node  ·  agents/",
            ha="center", va="center", fontsize=9, color=C_GRAY)
    ax.plot([0.5, 11.5], [12.88, 12.88], color=C_TEAL, lw=2)

    # ── GateNet ──
    _box(ax, 6.0, 12.4, 10.5, 0.72,
         "gate_node  (GateNet 라우팅)", C_BLUE, fontsize=12, bold=True,
         sub="후보별 전문가 가중치 산출  →  state: gate_weights_map  |  agents/gate_agent.py",
         sub_fg=C_GRAY)

    _arrow(ax, 6.0, 12.04, 6.0, 11.65)

    # ── 전문가 fan-out 헤더 ──
    ax.text(6.0, 11.52, "▼  5개 전문가 병렬 fan-out  (LangGraph)",
            ha="center", va="center", fontsize=9.5, color=C_TEAL, fontstyle="italic")

    # 5개 전문가 박스
    expert_defs = [
        (1.1,  "growth\n성장주 전문가",
         "고PER·실적 모멘텀\nop_income_yoy>20%\nRSI 50~70\nvolume_ratio>1.5"),
        (3.3,  "value\n가치주 전문가",
         "저PBR·자산가치\npbr<1 + roe>8%\ndebt<100%\nper_vs<-20%"),
        (5.5,  "theme\n테마주 전문가",
         "정책·이슈 모멘텀\nvolume_ratio>3\nRSI<70\n공시 확인"),
        (7.7,  "dividend\n배당주 전문가",
         "고배당·방어주\ndiv>3%\npayout<70%\nfcf>0"),
        (9.9,  "crisis\n위기종목 전문가",
         "과매도 반등\nret_5d<-15%\nrsi<30\n공매도 확인"),
    ]
    for ex, title, desc in expert_defs:
        _box(ax, ex, 9.9, 2.0, 2.8, title, C_BLUE, fontsize=10, bold=True,
             sub=desc, sub_fg=C_GRAY)
        _arrow(ax, ex, 11.3, ex, 11.3, color=C_MID_BLUE)
        ax.annotate("", xy=(ex, 11.3), xytext=(6.0, 11.38),
                    arrowprops=dict(arrowstyle="-|>", color=C_MID_BLUE,
                                    lw=1.3, mutation_scale=11), zorder=2)

    # Claude API 주석
    _box(ax, 6.0, 8.15, 11.0, 0.52,
         "Claude Sonnet 4.6 API  ·  각 종목 × 전문가별 JSON 응답",
         C_LIGHT, fg=C_DARK, fontsize=9.5,
         sub="signal: BUY|HOLD|SELL  ·  confidence: 0.0~1.0  ·  score: 1~10  ·  target_return",
         sub_fg=C_GRAY)

    # 전문가 → Claude
    for ex, _, _ in expert_defs:
        _arrow(ax, ex, 8.5, ex, 8.4, color=C_GRAY)
        ax.annotate("", xy=(6.0, 8.41), xytext=(ex, 8.5),
                    arrowprops=dict(arrowstyle="-|>", color=C_GRAY,
                                    lw=1.0, mutation_scale=9), zorder=2)

    # expert picks → aggregator
    for ex, _, _ in expert_defs:
        ax.annotate("", xy=(ex, 7.65), xytext=(6.0, 7.89),
                    arrowprops=dict(arrowstyle="-|>", color=C_MID_BLUE,
                                    lw=1.2, mutation_scale=10), zorder=2)
        _box(ax, ex, 7.3, 1.9, 0.62,
             "{expert}_picks", C_DARK, fontsize=8.5,
             sub="BUY 후보 리스트", sub_fg=C_TEAL)

    ax.text(6.0, 7.9, "▼  fan-in  →  aggregator",
            ha="center", va="center", fontsize=9, color=C_TEAL, fontstyle="italic")

    # ── Aggregator ──
    _arrow(ax, 6.0, 6.99, 6.0, 6.55)
    _box(ax, 6.0, 6.2, 10.5, 0.72,
         "aggregator_node", C_MID_BLUE, fontsize=12, bold=True,
         sub="BUY + confidence ≥ 0.60 필터  |  portfolio/aggregator.py",
         sub_fg=C_GRAY)

    # 복합 점수 공식
    _box(ax, 6.0, 5.15, 10.5, 0.9,
         "composite_score 산출", C_LIGHT, fg=C_DARK, fontsize=10, bold=True,
         sub="Σ(confidence × score × gate_weight) / n  +  len(전문가 집합) × 0.5",
         sub_fg=C_DARK)
    _arrow(ax, 6.0, 5.84, 6.0, 5.6)

    _arrow(ax, 6.0, 4.7, 6.0, 4.27)
    _box(ax, 6.0, 3.95, 10.5, 0.62,
         "TOP-5 포트폴리오 선정  (settings: top_n=5)",
         C_BLUE, fontsize=11, bold=True,
         sub="composite_score 내림차순 정렬  →  상위 5종목",
         sub_fg=C_GRAY)

    # ── Reporter ──
    _arrow(ax, 6.0, 3.64, 6.0, 3.22)
    _box(ax, 6.0, 2.9, 10.5, 0.62,
         "reporter_node", C_MID_BLUE, fontsize=11, bold=True,
         sub="텍스트 리포트 생성  ·  portfolio/aggregator.py",
         sub_fg=C_GRAY)

    # ── 출력 ──
    _arrow(ax, 6.0, 2.59, 6.0, 2.17)
    _box(ax, 6.0, 1.85, 10.5, 0.62,
         "reports/report_{date}.txt  +  portfolio_{date}.json",
         C_GREEN, fg=C_WHITE, fontsize=10, bold=True,
         sub="PortfolioStore(Supabase) 저장  →  다음 실행 시 수익률 평가",
         sub_fg=C_DARK)

    fig.tight_layout(pad=0.3)
    out = OUT_DIR / "03_종목평가_흐름도.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_WHITE)
    plt.close(fig)
    print(f"저장: {out}")


if __name__ == "__main__":
    make_db_build()
    make_screener()
    make_evaluation()
    print("완료.")
