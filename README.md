# Korean-MERA: 한국주식 AI 에이전트 시스템

> MERA (Mixture-of-Experts with Retrieval-Augmented representation) 논문 (WWW 2025) 기반의 한국 주식 신호 생성 AI 에이전트 시스템

## 개요

GPU 없이 외부 LLM API만으로 구현한 한국주식 포트폴리오 분석 시스템입니다.

- **임베딩**: KR-SBERT (로컬, 무료) — 한국어 특화 문장 임베딩
- **LLM**: Gemini 2.5 Flash API — 저비용 고성능 추론
- **벡터 DB**: ChromaDB — 유사 패턴 검색
- **데이터**: pykrx + FinanceDataReader (KOSPI200 + KOSDAQ150 = 400종목)

## 아키텍처

```
[데이터 수집] pykrx / FinanceDataReader
      ↓
[피처 엔지니어링] 기술적 지표 (RSI, MACD, BB, ADX 등)
      ↓
[임베딩] KR-SBERT → 패턴 텍스트 벡터화
      ↓
[ChromaDB] 유사 과거 패턴 검색 (top-k=5)
      ↓
[GateNet] Gemini → 전문가 에이전트 라우팅
      ↓
[Expert Agents] Gemini × 5종 (성장/가치/테마/배당/위기)
      ↓
[Aggregator] 신호 통합 → 포트폴리오 (TOP 20)
```

## 전문가 에이전트

| 에이전트 | 분석 관점 |
|----------|-----------|
| growth | 성장 모멘텀, 매출/이익 성장률, 섹터 성장성 |
| value | 저PBR/PER, 자산가치, 배당 수익률 |
| theme | 정책/이슈 수혜 테마, 사이클 위치 |
| dividend | 배당 안정성, 현금흐름, 주주환원 |
| crisis | 급락 반등, 과매도 기술적 반등 |

## 설치

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 환경 변수 설정
cp .env.example .env
# .env 파일에 API 키 입력:
#   GOOGLE_API_KEY=your_gemini_api_key
#   DART_API_KEY=your_dart_api_key (선택)
```

### API 키 발급

- **Gemini API**: [Google AI Studio](https://aistudio.google.com/app/apikey) → 무료 티어 제공
- **DART API**: [DART 오픈API](https://opendart.fss.or.kr/) → 공시 데이터 (선택사항)

## 사용법

```bash
# 1. 최초 실행: 히스토리 DB 구축 (3년치 패턴 저장, 1회만 실행)
py main.py --mode build_db --years 3

# 2. 일별 신호 생성 (오늘 날짜)
py main.py --mode run

# 3. 특정 날짜 분석
py main.py --mode run --date 20260515

# 4. 스케줄러 실행 (매일 16:30 자동 실행)
py main.py --mode schedule

# 5. 백테스트
py main.py --mode backtest
```

### 출력 파일

```
reports/
  report_20260515.txt     # 포트폴리오 리포트 (텍스트)
  portfolio_20260515.csv  # 포트폴리오 데이터 (CSV)
```

## 성능

- **처리 속도**: 400종목 약 7분 (병렬 workers=10)
- **API 비용**: Gemini Flash 기준 약 400종목 × 2회 = 800 API 호출/일
- **DB 구축**: KOSPI200+KOSDAQ150 × 3년 ≈ 6,000~7,000 패턴

## 프로젝트 구조

```
korean-mera/
├── main.py                 # 진입점 (CLI)
├── config/
│   ├── __init__.py         # 설정 로딩 (API 키 등)
│   └── settings.yaml       # 시스템 파라미터
├── data/
│   ├── collector.py        # OHLCV 수집 (pykrx + FDR)
│   ├── feature_engineer.py # 기술적 지표 계산
│   └── text_generator.py   # 패턴 텍스트 생성
├── vector_store/
│   ├── embedder.py         # KR-SBERT 임베딩
│   └── store.py            # ChromaDB CRUD
├── agents/
│   ├── base.py             # Gemini API 공통 호출
│   ├── gate_agent.py       # GateNet 라우팅
│   └── experts.py          # 5개 전문가 에이전트
├── portfolio/
│   └── aggregator.py       # 신호 통합 및 포트폴리오 구성
├── scheduler/
│   ├── pipeline.py         # 메인 파이프라인 (병렬 처리)
│   └── daily_runner.py     # APScheduler 자동화
└── evaluation/
    └── backtest.py         # 백테스트
```

## 권장 일과

```bash
# 매일 장 마감 후 (16:30~)
py main.py --mode run          # 신호 생성
py main.py --mode update_today # ChromaDB 오늘 패턴 추가 (선택)
```

## 참고 논문

- MERA: Multi-hop Event Reasoning and Abstraction (WWW 2025)
- 한국 주식시장 적용을 위해 KR-SBERT 임베딩 및 한국어 프롬프트로 재구성
