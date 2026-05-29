# Korean-MERA: 한국주식 AI 에이전트 시스템

> MERA (Mixture-of-Experts with Retrieval-Augmented representation) 논문 (WWW 2025) 기반의 한국 주식 신호 생성 AI 에이전트 시스템

## 개요

GPU 없이 외부 LLM API만으로 구현한 한국주식 포트폴리오 분석 시스템입니다.

- **임베딩**: OpenAI `text-embedding-3-small` (1,536차원) — API 기반
- **LLM**: Anthropic Claude `claude-sonnet-4-6` — 전문가 에이전트 추론
- **벡터 DB**: ChromaDB — 유사 과거 패턴 검색
- **데이터**: pykrx + FinanceDataReader + 네이버 금융 API (KOSPI200, 200종목)

---

## 목차

1. [사전 요구사항](#사전-요구사항)
2. [설치](#설치)
3. [환경 변수 설정](#환경-변수-설정)
4. [최초 실행 (DB 구축)](#최초-실행-db-구축)
5. [일별 실행](#일별-실행)
6. [백테스트](#백테스트)
7. [대시보드](#대시보드)
8. [유틸리티 스크립트](#유틸리티-스크립트)
9. [테스트](#테스트)
10. [아키텍처](#아키텍처)
11. [프로젝트 구조](#프로젝트-구조)

---

## 사전 요구사항

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | **3.13** | pyenv로 자동 설치 가능 |
| Poetry | **1.8+** | 의존성 관리 |
| pyenv | 최신 | Python 버전 관리 |
| API 키 | Anthropic, OpenAI | 필수 (아래 참고) |

### API 키 발급

| 서비스 | 발급 링크 | 용도 |
|--------|-----------|------|
| Anthropic | https://console.anthropic.com | LLM 추론 **(필수)** |
| OpenAI | https://platform.openai.com | 임베딩 **(필수)** |
| KRX | https://open.krx.co.kr | 시장 데이터 **(필수)** |
| DART | https://opendart.fss.or.kr | 공시 데이터 (선택) |
| Supabase | https://supabase.com | 포트폴리오 이력 저장 (선택) |

---

## 설치

### 1. pyenv 설치

**Windows (PowerShell 관리자 권한)**
```powershell
# 실행 정책 허용 (최초 1회)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# pyenv-win 설치
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

설치 후 PowerShell을 새로 열고 확인:
```powershell
pyenv --version
```

> **자동 설치 후에도 `pyenv`를 찾지 못할 경우** — 환경변수가 등록되지 않은 것입니다. 아래 명령을 PowerShell에서 실행하세요 (관리자 권한 불필요):

```powershell
# 환경변수 등록 (User 범위)
[System.Environment]::SetEnvironmentVariable('PYENV',      "$env:USERPROFILE\.pyenv\pyenv-win\",       'User')
[System.Environment]::SetEnvironmentVariable('PYENV_ROOT', "$env:USERPROFILE\.pyenv\pyenv-win\",       'User')
[System.Environment]::SetEnvironmentVariable('PYENV_HOME', "$env:USERPROFILE\.pyenv\pyenv-win\",       'User')

# PATH에 bin / shims 추가
$cur = [System.Environment]::GetEnvironmentVariable('PATH', 'User')
[System.Environment]::SetEnvironmentVariable('PATH',
    "$env:USERPROFILE\.pyenv\pyenv-win\bin;$env:USERPROFILE\.pyenv\pyenv-win\shims;$cur", 'User')
```

설정 후 PowerShell을 **새로 열면** `pyenv --version`이 정상 동작합니다.

**Linux / macOS**
```bash
curl https://pyenv.run | bash
# ~/.bashrc 또는 ~/.zshrc에 아래 추가:
# export PYENV_ROOT="$HOME/.pyenv"
# export PATH="$PYENV_ROOT/bin:$PATH"
# eval "$(pyenv init -)"
```

### 2. Python 3.13 설치

```bash
pyenv install 3.13.0
```

프로젝트 디렉토리에 `.python-version` 파일이 있으므로 해당 디렉토리에서는 자동으로 3.13.0이 활성화됩니다.

### 3. Poetry 설치

```bash
# Linux / macOS
curl -sSL https://install.python-poetry.org | python3 -

# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 4. 저장소 클론

```bash
git clone https://github.com/ssu-fin-ai/korean-mera.git
cd korean-mera
```

Python 버전을 고정하고 Poetry 가상환경을 프로젝트 내부에 생성하도록 설정합니다.

```bash
pyenv local 3.13.0
poetry config virtualenvs.in-project true
```

### 5. 의존성 설치

```bash
poetry install
```

`poetry.lock`이 있으면 고정된 버전으로 설치되고, 없으면 `pyproject.toml` 범위 내 최신 버전으로 설치됩니다.

---

## 환경 변수 설정

`.env.example`을 복사해 `.env` 파일을 만듭니다.

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 항목을 입력합니다.

```dotenv
# 필수
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
KRX_ID=your_krx_id
KRX_PW=your_krx_password

# 선택: 공시 데이터 (미설정 시 공시 없이 동작)
DART_API_KEY=...

# 선택: 포트폴리오 이력 저장 (미설정 시 로컬 파일만 사용)
SUPABASE_URL=http://your-supabase-host:8000
SUPABASE_KEY=...
```

> `.env` 파일은 `.gitignore`에 포함되어 있으므로 절대 커밋되지 않습니다.

---

## 실행 방법

모든 명령은 `poetry run`을 앞에 붙여 실행합니다.

### 1단계: DB 구축 (최초 1회)

ChromaDB에 3년치 과거 패턴을 임베딩하여 저장합니다.

```bash
poetry run py main.py --mode build_db --years 3
```

- 소요 시간: 약 140분 (일별 샘플링 기준)
- 결과물: `chroma_db/` 디렉토리에 약 150,000건 패턴 저장
- 비용: OpenAI 임베딩 기준 약 $0.36

### 2단계: 일별 실행

```bash
# 오늘 날짜 분석 (기본: 월별 분석, 20일 RAG)
poetry run py main.py --mode run

# 특정 날짜 분석
poetry run py main.py --mode run --date 20260515

# 주간 분석 (5일 RAG 기준)
poetry run py main.py --mode run --date 20260515 --horizon weekly

# 월별 분석 (20일 RAG 기준, 기본값)
poetry run py main.py --mode run --date 20260515 --horizon monthly
```

| `--horizon` | RAG 수익률 기준 | 적합한 용도 |
|-------------|----------------|------------|
| `weekly` | 5일 후 수익률 | 단기 모멘텀·테마 분석 |
| `monthly` | 20일 후 수익률 (기본값) | 중장기 가치·배당 분석 |

### 포트폴리오 평가

```bash
poetry run py main.py --mode evaluate --date 20260523
```

### 백테스트 (ARR·MDD·Sortino·Calmar)

```bash
poetry run py main.py --mode backtest
```

### 스케줄러 (매일 16:30 자동화)

```bash
poetry run py main.py --mode schedule
```

### 출력 파일

```
reports/
  report_20260515.txt        # 포트폴리오 리포트 (텍스트)
  portfolio_20260515.csv     # 포트폴리오 데이터 (CSV)

cache/
  {ticker}_{start}_{end}.parquet  # OHLCV 캐시
  fin_{ticker}_{YYYYMM}.json      # 재무 지표 캐시 (월별)
  sector_map.parquet              # 섹터 정보 캐시 (주별)
```

---

## 백테스트

### 2026년 백테스트

```bash
# 월별 + 주별 (기본값)
py backtest_2026.py

# 월별만 (5회)
py backtest_2026.py --freq monthly

# 주별만 (~20회)
py backtest_2026.py --freq weekly

# OHLCV 캐시 단계 건너뜀 (이미 캐시된 경우)
py backtest_2026.py --freq monthly --no-precache
```

결과물은 `reports/backtest_2026/` 에 날짜별 디렉토리로 저장됩니다.

```
reports/backtest_2026/
  20260102/
    combined_portfolio.json   # 통합 포트폴리오
    growth_picks.json         # 성장주 전문가 선정
    value_picks.json
    theme_picks.json
    dividend_picks.json
    crisis_picks.json
    eval_20260130.json        # 수익률 평가 결과
    report.txt
  summary_monthly.txt         # 월별 종합 리포트
  summary_weekly.txt          # 주별 종합 리포트
  run_log.txt
```

### 2025년 백테스트

```bash
# 월별 + 주별 (기본값)
py backtest_2025.py

# 월별만
py backtest_2025.py --freq monthly

# 주별만
py backtest_2025.py --freq weekly
```

결과물은 `reports/backtest_2025/` 에 동일한 구조로 저장됩니다.

### Supabase 업로드 (선택)

백테스트 결과를 Supabase DB에 업로드합니다. SUPABASE_URL, SUPABASE_KEY 설정이 필요합니다.

```bash
py upload_backtest_2026.py
py upload_backtest_2025.py
```

---

## 대시보드

### 실행

```bash
streamlit run app.py
```

브라우저에서 `http://localhost:8501` 접속

### 백그라운드 실행 (Linux)

```bash
# 시작
bash start.sh

# 종료
bash stop.sh

# 로그 확인
tail -f streamlit.log
```

### 페이지 구성

| 페이지 | 내용 |
|--------|------|
| 포트폴리오 대시보드 | 최근 포트폴리오 이력·신호 현황 |
| 백테스트 2026 | 2026년 월별/주별 성과 (ARR, MDD, Sortino, Calmar) |
| 백테스트 2025 | 2025년 월별/주별 성과 + 전문가별 비교 |

---

## 유틸리티 스크립트

### 어그리게이터 재실행

포트폴리오 선정 단계는 건너뛰고 집계(aggregator) 단계만 다시 실행합니다.

```bash
py rerun_aggregator.py --date 20260515
```

---

## 테스트

### 연결 확인 (권장: 최초 설정 후 실행)

```bash
# API 키 및 데이터 수집 전체 흐름 확인
py test_smoke.py

# Claude LLM 연결 및 JSON 파싱 확인
py test_llm_parse.py

# JSON 파싱 엣지케이스 확인
py test_json_parse.py
```

---

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

### 전문가 에이전트

| 에이전트 | 전문 분야 | 주요 신호 |
|----------|-----------|-----------|
| growth | 성장주 | RSI 60+, MACD 골든크로스, 거래량 급증, 포워드PER 개선 |
| value | 가치주 | PBR 1 미만, 배당수익률 3%+, 저변동성 |
| theme | 테마주 | 거래량 3배+, 공시 이슈, 볼린저 상단 돌파 |
| dividend | 배당주 | 배당수익률 4%+, 방어 섹터, 이익 안정 |
| crisis | 위기종목 | 5일 수익률 -10% 이하, RSI 과매도, 일시 악재 |

### 성능

| 항목 | 수치 |
|------|------|
| 처리 속도 | KOSPI200 (200종목) 약 7분 (workers=10) |
| Claude API 호출 | 약 200종목 × 2~3회 = 400~600회/일 |
| OpenAI 임베딩 비용 | build_db 1회 기준 ~$0.03 (5,972건) |
| ChromaDB 규모 | 5,972건 패턴 (KOSPI200 × 3년, 월말 샘플링) |

---

## 프로젝트 구조

```
korean-mera/
├── main.py                   # 진입점 (CLI)
├── app.py                    # Streamlit 대시보드
├── backtest_2025.py          # 2025년 백테스트
├── backtest_2026.py          # 2026년 백테스트
├── rerun_aggregator.py       # 어그리게이터 단독 재실행
├── upload_backtest_2025.py   # 2025 결과 Supabase 업로드
├── upload_backtest_2026.py   # 2026 결과 Supabase 업로드
├── config/
│   ├── __init__.py           # API 키·설정 로딩
│   └── settings.yaml         # 시스템 파라미터
├── data/
│   ├── collector.py          # OHLCV·재무·공시 수집
│   ├── feature_engineer.py   # 기술적 지표 계산
│   └── text_generator.py     # 패턴 텍스트 생성
├── vector_store/
│   ├── embedder.py           # OpenAI 임베딩 (1,536차원)
│   └── store.py              # ChromaDB CRUD
├── agents/
│   ├── base.py               # Claude API 공통 호출
│   ├── gate_agent.py         # GateNet 라우팅
│   ├── experts.py            # 5개 전문가 에이전트
│   ├── graph.py              # LangGraph 파이프라인
│   ├── screener.py           # 종목 스크리너
│   └── state.py              # LangGraph 상태 정의
├── portfolio/
│   ├── aggregator.py         # 신호 통합 및 포트폴리오 구성
│   └── evaluator.py          # 실적 평가·리스크 지표 산출
├── scheduler/
│   ├── pipeline.py           # 메인 파이프라인 (병렬 처리)
│   └── daily_runner.py       # APScheduler 자동화
├── evaluation/
│   └── backtest.py           # 백테스트 (ARR·MDD·Sortino·Calmar)
├── db/
│   └── supabase_store.py     # 포트폴리오·평가 이력 저장
├── pages/
│   ├── 포트폴리오_대시보드.py
│   ├── 2_백테스트_2026.py
│   └── 3_백테스트_2025.py
├── reports/                  # 실행 결과 출력
├── chroma_db/                # 메인 벡터 DB (OpenAI 임베딩)
├── chroma_db_gemini/         # 구버전 백업 (KR-SBERT 768차원)
├── cache/                    # OHLCV·재무 캐시 (.gitignore)
└── docs/
    ├── data_guide.md
    └── indicator_review.md
```

---

## 알려진 이슈

- **pykrx 주말/공휴일 실패** → FinanceDataReader로 자동 fallback
- **KRX 로그인 실패** → `.env`에 `KRX_ID` / `KRX_PW` 설정 필요 (https://open.krx.co.kr 가입)
- **Claude 응답 마크다운 코드블록** → `parse_json_response()`가 자동 처리

---

## 참고 논문

- MERA: Multi-hop Event Reasoning and Abstraction (WWW 2025)
- 한국 주식시장 적용: OpenAI 임베딩 + Claude 추론 + 네이버 금융 재무 데이터로 재구성
