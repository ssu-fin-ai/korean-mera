# Korean-MERA: 한국주식 AI 에이전트 시스템

> MERA (Mixture-of-Experts with Retrieval-Augmented representation) 논문 (WWW 2025) 기반의 한국 주식 신호 생성 AI 에이전트 시스템

## 개요

GPU 없이 외부 LLM API만으로 구현한 한국주식 포트폴리오 분석 시스템입니다.

- **임베딩**: OpenAI `text-embedding-3-small` (1,536차원) — API 기반
- **LLM**: Anthropic Claude `claude-sonnet-4-6` — 전문가 에이전트 추론
- **벡터 DB**: ChromaDB — 유사 과거 패턴 검색
- **데이터**: pykrx + FinanceDataReader + 네이버 금융 API (KOSPI200, 200종목)

## 아키텍처

```
[데이터 수집] pykrx / FinanceDataReader / 네이버 금융 API
      ↓
[피처 엔지니어링] 기술적 지표 (RSI, MACD, BB, ADX 등) + 재무 지표 (PER, PBR 등)
      ↓
[임베딩] OpenAI text-embedding-3-small → 패턴 텍스트 벡터화 (1,536차원)
      ↓
[ChromaDB] 유사 과거 패턴 검색 (top-k=5, 코사인 유사도)
      ↓
[GateNet] Claude → 전문가 에이전트 라우팅
      ↓
[Expert Agents] Claude × 5종 (성장/가치/테마/배당/위기)
      ↓
[Aggregator] 신호 통합 → 포트폴리오 (TOP 20, 신뢰도 ≥ 0.60)
      ↓
[Evaluator] 실제 수익률 평가 + ARR/MDD/Sortino/Calmar 리스크 지표 산출
```

## 데이터 파이프라인

### 수집 데이터

| 구분 | 소스 | 항목 | 갱신 주기 |
|------|------|------|-----------|
| 주가 OHLCV | pykrx / FDR | 시가·고가·저가·종가·거래량·거래대금 | 일별 (캐시) |
| 지수 | pykrx / FDR (KS11) | KOSPI 지수 OHLCV | 일별 |
| 섹터 정보 | FinanceDataReader | 종목명·섹터·시장 구분 | 주별 캐시 |
| 재무 지표 | 네이버 금융 API | PER·PBR·EPS·BPS·배당수익률·컨센서스 PER/EPS | 월별 캐시 |
| 공시 데이터 | DART API | 최근 30일 공시 제목·날짜 | 실시간 (선택) |

### 기술적 지표 (Feature Engineering)

| 지표 | 계산 방법 | 의미 |
|------|-----------|------|
| RSI (14일) | Wilder smoothing | 과매수(70+) / 과매도(30-) |
| MACD diff | EMA12 - EMA26, Signal=EMA9 | 골든크로스 / 데드크로스 |
| 볼린저밴드 %B | (종가 - 하단) / (상단 - 하단) | 밴드 내 상대 위치 |
| ADX | Average Directional Index | 추세 강도 (25+ = 강한 추세) |
| MFI | Money Flow Index (14일) | 거래량 반영 RSI |
| 거래량 비율 | 당일 거래량 / 20일 평균 | 이상 거래량 감지 |
| 역사적 변동성 | 20일 일간 수익률 표준편차 | 변동성 수준 |
| MA20/MA60 대비 | (종가 / MA - 1) | 단기·중기 추세 |
| 5일/20일 수익률 | 단순 수익률 | 단기 모멘텀 |
| 상대강도 | 종목 수익률 - KOSPI 수익률 | 시장 대비 초과 성과 |

### 재무 지표 (네이버 금융, 월별 캐시)

| 지표 | 의미 | 활용 전문가 |
|------|------|------------|
| PER | 주가 / EPS (주가수익비율) | value, growth |
| PBR | 주가 / BPS (주가순자산비율) | value |
| EPS | 주당순이익 | growth, value |
| BPS | 주당순자산 | value |
| 배당수익률 | 연간 배당금 / 주가 | dividend |
| 컨센서스 PER | 증권사 예상 PER | growth |
| 컨센서스 EPS | 증권사 예상 EPS | growth |

### ChromaDB 패턴 저장소

- **구축 방식**: 월말 날짜 샘플링 (종목당 ~36개 스냅샷 / 3년)
- **현재 규모**: KOSPI200 × 3년 = **5,972건** 패턴
- **임베딩 차원**: 1,536 (OpenAI text-embedding-3-small)
- **유사도 측정**: 코사인 유사도
- **백업**: `chroma_db_gemini/` — KR-SBERT 768차원 구버전

## 전문가 에이전트 (Claude Sonnet)

### GateNet — 전문가 라우팅

모든 종목은 먼저 GateNet을 통과합니다. GateNet은 현재 패턴과 ChromaDB에서 검색한 유사 과거 5개 사례를 함께 분석해 **1~2개의 전문가를 자동 선택**합니다.

```
입력: 현재 기술적 패턴 텍스트 + 유사 과거 패턴 5개
출력: {"experts": ["growth"], "pattern_type": "실적모멘텀 성장주", "confidence": 0.82}
```

---

### growth — 성장주 전문가

**언제 선택되나**: RSI 강세(60+), MACD 골든크로스, 거래량 증가, 20일 수익률 플러스 추세

| 판단 기준 | 참고 데이터 | 세부 내용 |
|-----------|------------|-----------|
| 실적 모멘텀 | 포워드PER, 컨센서스EPS | 트레일링PER 대비 포워드PER이 낮을수록 실적 회복 기대 |
| 주가 모멘텀 | RSI, MACD, 거래량 비율 | RSI 60~70 + MACD 골든크로스 + 거래량 2배 = 강한 매수 신호 |
| 섹터 성장성 | 섹터 정보, 공시 | AI·반도체·2차전지·바이오 섹터 가중치 부여 |
| 리스크 | 볼린저밴드 %B, RSI | 과매수 구간(RSI 70+, BB 상단 돌파) 경고 |

**주요 판단**: 포워드PER이 낮고 컨센서스EPS가 실제EPS보다 훨씬 높으면 실적 회복 기대 → BUY 가중

---

### value — 가치주 전문가

**언제 선택되나**: 저PBR, 낮은 변동성, MA 하단 횡보, 거래량 정상

| 판단 기준 | 참고 데이터 | 세부 내용 |
|-----------|------------|-----------|
| 밸류에이션 | PBR, PER, BPS | PBR 1 미만 = 자산가치 이하, PER 섹터 평균 대비 할인율 |
| 재무 안정성 | 영업이익, 순이익 (DART) | 이익 창출 능력, 부채 상환 여력 |
| 배당·자본환원 | 배당수익률, DPS | 꾸준한 배당 = 현금흐름 안정성 증거 |
| 촉매 여부 | 공시, 뉴스 | 저평가 해소를 위한 이벤트(자사주 매입, 실적 서프라이즈) |

**주요 판단**: PBR 1 미만 + 배당수익률 3%+ + 이익 안정 → BUY / 촉매 없이 장기 횡보 → HOLD

---

### theme — 테마주 전문가

**언제 선택되나**: 거래량 폭발(3배+), 단기 급등, 공시 이슈, 볼린저 상단 돌파

| 판단 기준 | 참고 데이터 | 세부 내용 |
|-----------|------------|-----------|
| 이슈 강도 | 공시 제목, 뉴스 | 정책 직접 수혜 vs 간접 수혜 구분 |
| 수급 과열 | RSI, 거래량 비율 | RSI 70+ + 거래량 3배+ = 단기 과열 경고 |
| 테마 사이클 | 유사 과거 패턴 | 초기(매수)/중기(보유)/말기(매도) 위치 판단 |
| 손절선 | 볼린저밴드, 지지선 | 테마주 특성상 명확한 손절 기준 제시 필수 |

**주요 판단**: 테마 초기 + 직접 수혜 + 거래량 급증 → BUY / 테마 말기 + 과열 + 이슈 소멸 → SELL

---

### dividend — 배당주 전문가

**언제 선택되나**: 저변동성, 방어 섹터(통신·유틸리티·금융), 꾸준한 수익률

| 판단 기준 | 참고 데이터 | 세부 내용 |
|-----------|------------|-----------|
| 배당 매력 | 배당수익률, DPS, BPS | 시가배당률 3%+ = 채권 대비 매력도 평가 |
| 배당 지속성 | 영업이익, 순이익 (DART) | 이익 대비 배당성향, 미래 배당 유지 가능성 |
| 방어적 특성 | 역사적 변동성, 상대강도 | 시장 하락 시 KOSPI 대비 낙폭 축소 여부 |
| 금리 환경 | 섹터 정보 | 금리 인상기 = 채권 경쟁 → 배당주 매력 감소 |

**주요 판단**: 배당수익률 4%+ + 이익 안정 + 저변동성 → BUY / 이익 감소로 배당 삭감 우려 → SELL

---

### crisis — 위기종목 전문가

**언제 선택되나**: 급락(5일 수익률 -10% 이하), 볼린저 하단 접근, RSI 과매도(30-)

| 판단 기준 | 참고 데이터 | 세부 내용 |
|-----------|------------|-----------|
| 급락 원인 | 공시, 뉴스, 재무 | 일시적 이슈(악재 과반응) vs 구조적 문제(실적 악화) |
| 반등 가능성 | MFI, RSI, 유사 급락 사례 | 과거 동일 패턴 급락 후 평균 회복률 참조 |
| 지지선 | 볼린저밴드, MA60 | 기술적 지지 구간에서의 반등 가능성 |
| 손절 기준 | 트레일링PER, PBR | PBR 1 미만이면 자산가치 지지, 초과 손실 시 손절 |

**주요 판단**: 일시적 악재 + MFI 과매도 + 유사패턴 반등 사례 → BUY / 구조적 실적 악화 + 회복 근거 없음 → SELL

---

### 공통 입력 데이터 (모든 전문가)

```
현재 종목 패턴 텍스트
├── 기술적 지표: RSI, MACD, 볼린저밴드, ADX, MFI, 거래량비율, MA대비, 변동성
├── 수익률: 5일/20일 수익률, KOSPI 대비 상대강도
└── 재무 지표: 포워드PER, 트레일링PER, PBR, EPS, 컨센서스EPS, BPS, 배당수익률

유사 과거 패턴 5개 (ChromaDB RAG)
└── 비슷한 기술적 셋업의 과거 사례 + 당시 5일 후 실제 수익률

최근 공시/뉴스 (DART, 선택)
└── 최근 30일 공시 제목 목록
```

## 성과 지표 (Backtest / Evaluation)

### 수익률 지표

| 지표 | 산식 | 의미 |
|------|------|------|
| 평균 수익률 | 보유기간 수익률 평균 | 개별 신호 평균 성과 |
| 적중률 (Hit Rate) | 수익 신호 수 / 전체 신호 수 | 방향 예측 정확도 |
| ARR (연환산 수익률) | (누적수익)^(252/보유일수×신호수) - 1 | 연간 기준 수익률 |

### 리스크 지표

| 지표 | 산식 | 의미 |
|------|------|------|
| MDD (최대 낙폭) | min((누적 - 고점) / 고점) | 가장 큰 손실 구간 |
| 샤프 지수 (연환산) | (평균수익 / 표준편차) × √(252/보유일) | 단위 리스크당 수익 |
| 소르티노 지수 | (평균수익 / 하방표준편차) × √(252/보유일) | 손실 구간만 반영한 리스크 조정 수익 |
| 칼마 지수 | ARR / \|MDD\| | 최대 낙폭 대비 연수익 |

> **보유기간**: 기본 5일 (주간 리밸런싱), `--hold_days` 옵션으로 변경 가능

## 설치

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
```

`.env` 파일에 API 키 입력:

```
ANTHROPIC_API_KEY=...   # 필수: Claude LLM
OPENAI_API_KEY=...      # 필수: 임베딩
DART_API_KEY=...        # 선택: 공시 데이터
SUPABASE_URL=...        # 선택: 포트폴리오 이력 저장
SUPABASE_KEY=...        # 선택: 위와 동일
```

### API 키 발급

| 서비스 | 링크 | 용도 |
|--------|------|------|
| Anthropic | [console.anthropic.com](https://console.anthropic.com) | LLM 추론 (필수) |
| OpenAI | [platform.openai.com](https://platform.openai.com) | 임베딩 (필수) |
| DART | [opendart.fss.or.kr](https://opendart.fss.or.kr) | 공시 데이터 (선택) |

## 사용법

```bash
# 1. 최초 실행: 히스토리 DB 구축 (3년치 패턴, 1회만 실행)
py main.py --mode build_db --years 3

# 2. 일별 신호 생성 (오늘 날짜)
py main.py --mode run

# 3. 특정 날짜 분석
py main.py --mode run --date 20260515

# 4. 스케줄러 실행 (매일 16:30 자동 실행)
py main.py --mode schedule

# 5. 백테스트 (수익률 분포·MDD·ARR 차트 생성)
py main.py --mode backtest

# 6. 포트폴리오 평가 (전주 포트폴리오 실적 분석)
py main.py --mode evaluate --date 20260523
```

### 출력 파일

```
reports/
  report_20260515.txt        # 포트폴리오 리포트 (텍스트)
  portfolio_20260515.csv     # 포트폴리오 데이터 (CSV)
  backtest_result.png        # 백테스트 차트 (2×3, ARR/MDD/Sortino 포함)

cache/
  {ticker}_{start}_{end}.parquet  # OHLCV 캐시
  fin_{ticker}_{YYYYMM}.json      # 재무 지표 캐시 (월별)
  sector_map.parquet              # 섹터 정보 캐시 (주별)
```

### 대시보드

```bash
streamlit run app.py
```

포트폴리오 이력·평가 결과·ARR/MDD/Sortino/Calmar 지표를 웹 UI로 조회

## 성능

| 항목 | 수치 |
|------|------|
| 처리 속도 | KOSPI200 (200종목) 약 7분 (workers=10) |
| Claude API 호출 | 약 200종목 × 2~3회 = 400~600회/일 |
| OpenAI 임베딩 비용 | build_db 1회 기준 ~$0.03 (5,972건) |
| ChromaDB 규모 | 5,972건 패턴 (KOSPI200 × 3년, 월말 샘플링) |

## 프로젝트 구조

```
korean-mera/
├── main.py                 # 진입점 (CLI)
├── app.py                  # Streamlit 대시보드
├── config/
│   ├── __init__.py         # API 키·설정 로딩
│   └── settings.yaml       # 시스템 파라미터
├── data/
│   ├── collector.py        # OHLCV·재무·공시 수집
│   ├── feature_engineer.py # 기술적 지표 계산
│   └── text_generator.py   # 패턴 텍스트 생성
├── vector_store/
│   ├── embedder.py         # OpenAI 임베딩 (1,536차원)
│   └── store.py            # ChromaDB CRUD
├── agents/
│   ├── base.py             # Claude API 공통 호출
│   ├── gate_agent.py       # GateNet 라우팅
│   └── experts.py          # 5개 전문가 에이전트
├── portfolio/
│   ├── aggregator.py       # 신호 통합 및 포트폴리오 구성
│   └── evaluator.py        # 실적 평가·리스크 지표 산출
├── scheduler/
│   ├── pipeline.py         # 메인 파이프라인 (병렬 처리)
│   └── daily_runner.py     # APScheduler 자동화
├── evaluation/
│   └── backtest.py         # 백테스트 (ARR·MDD·Sortino·Calmar)
├── db/
│   └── supabase_store.py   # 포트폴리오·평가 이력 저장
└── chroma_db_gemini/       # KR-SBERT 768차원 구버전 백업
```

## 권장 일과

```bash
# 매일 장 마감 후 (16:30~)
py main.py --mode run          # 신호 생성 + 전주 포트폴리오 자동 평가
```

## 참고 논문

- MERA: Multi-hop Event Reasoning and Abstraction (WWW 2025)
- 한국 주식시장 적용: OpenAI 임베딩 + Claude 추론 + 네이버 금융 재무 데이터로 재구성
