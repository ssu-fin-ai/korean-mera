# CLAUDE.md — Korean-MERA 프로젝트 가이드

## 프로젝트 개요

MERA 논문 기반 한국주식 AI 에이전트. GPU 없이 Gemini Flash + KR-SBERT로 동작.

- 브랜치: `feat/gemini-flash-kr-sbert`
- Python: 3.13 (Windows, `py` 명령어 사용)
- LLM: `gemini-2.5-flash` (thinking 비활성화 — `thinking_budget=0`)
- 임베딩: `snunlp/KR-SBERT-V40K-klueNLI-augSTS` (로컬)

## 핵심 명령어

```bash
# 테스트
py test_smoke.py

# DB 구축 (최초 1회)
py main.py --mode build_db --years 3

# 일별 실행
py main.py --mode run --date 20260515

# LLM 연결 테스트
py test_llm_parse.py
```

## 아키텍처 흐름

```
collector.py → feature_engineer.py → embedder.py → store.py (ChromaDB)
                                          ↓
                              gate_agent.py (GateNet 라우팅)
                                          ↓
                              experts.py (5개 전문가 에이전트)
                                          ↓
                              aggregator.py → pipeline.py → report
```

## 주요 설정 (`config/settings.yaml`)

| 설정 | 값 | 비고 |
|------|----|------|
| `llm.model` | `gemini-2.5-flash` | thinking_budget=0으로 속도 최적화 |
| `llm.max_tokens` | `8192` | JSON 잘림 방지 |
| `pipeline.workers` | `10` | ThreadPoolExecutor 병렬 처리 |
| `pipeline.api_concurrency` | `5` | Gemini 동시 호출 수 |
| `portfolio.top_n` | `20` | 최종 BUY 종목 수 |
| `portfolio.signal_threshold` | `0.60` | BUY 최소 신뢰도 |

## 환경 변수 (`.env`)

```
GOOGLE_API_KEY=...   # 필수: Gemini API
DART_API_KEY=...     # 선택: DART 공시 데이터
```

## 데이터 파이프라인 주의사항

- **pykrx는 주말/공휴일에 실패** → FDR 자동 fallback
- **캐시 키**: `cache/{ticker}_{start}_{end}.parquet` — 날짜 바뀌면 재다운로드
- `feature_engineer.compute()`: `label_5d/10d/20d` 컬럼은 `dropna()` 제외 (최근 20일 행 소실 방지)
- `get_snapshot_vector()`: 정확한 날짜 없으면 가장 가까운 이전 거래일 사용

## JSON 파싱

Gemini는 마크다운 코드블록(` ```json `)으로 응답. `parse_json_response()`가 처리:
1. ` ```json...``` ` 블록 추출
2. `_extract_json_object()` — 중괄호 깊이 추적으로 완전한 JSON 추출
3. trailing comma 자동 정리 후 재시도

## 성능

- 400종목 처리: **약 7분** (workers=10, thinking 비활성화)
- thinking 활성화 시: 응답당 20~45초 → 사용 금지

## 파일 역할 요약

| 파일 | 역할 |
|------|------|
| `agents/base.py` | Gemini API 호출, JSON 파싱 |
| `agents/gate_agent.py` | GateNet: 어떤 전문가 에이전트 쓸지 결정 |
| `agents/experts.py` | growth/value/theme/dividend/crisis 5개 에이전트 |
| `data/collector.py` | OHLCV 수집 (pykrx fallback → FDR) |
| `data/feature_engineer.py` | RSI, MACD, BB, ADX 등 기술적 지표 |
| `scheduler/pipeline.py` | 메인 파이프라인 (ThreadPoolExecutor 병렬) |
| `portfolio/aggregator.py` | 신호 통합, TOP-N 포트폴리오 구성 |
| `vector_store/store.py` | ChromaDB upsert/query |

## 알려진 이슈

- **pykrx 지수 컬럼**: 한글 컬럼명을 `_get_index_pykrx()`에서 영문으로 rename
- **pykrx 종목 컬럼 수**: 6~7개 가변 → 동적 처리
- **KRX 로그인 오류**: 정상 (환경변수 KRX_ID/KRX_PW 불필요, FDR fallback 동작)
