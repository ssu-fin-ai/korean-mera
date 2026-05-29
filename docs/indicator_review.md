# 지표 계산 검토 보고서

작성일: 2026-05-24  
검토 대상: `data/feature_engineer.py`, `data/collector.py`, `agents/screener.py`, `portfolio/aggregator.py`, `evaluation/backtest.py`

---

## 요약

| 심각도 | 건수 | 내용 |
|--------|------|------|
| 🔴 버그 | 1 | MACD 컬럼 인덱스 오류 — signal/histogram 뒤바뀜 |
| 🟡 비표준 | 2 | BBands 기간 5일(표준 20일), 상대강도 계산 방식 불일치 |
| 🟢 정상 | 나머지 | 수익률·모멘텀·변동성·재무비율·백테스트 성과지표 모두 적절 |

---

## 1. 수익률 지표 (`_add_returns`)

### 계산식

| 피처 | 계산식 | 평가 |
|------|--------|------|
| `ret_1d` | `close.pct_change()` | ✅ 정상 |
| `ret_3d` | `close.pct_change(3)` | ✅ 정상 |
| `ret_5d` | `close.pct_change(5)` | ✅ 정상 |
| `ret_10d` | `close.pct_change(10)` | ✅ 정상 |
| `ret_20d` | `close.pct_change(20)` | ✅ 정상 |
| `ret_60d` | `close.pct_change(60)` | ✅ 정상 |

### 레이블 (미래 수익률)

```python
label_5d  = close.pct_change(5).shift(-5)
label_10d = close.pct_change(10).shift(-10)
label_20d = close.pct_change(20).shift(-20)
```

**검증:**
- `pct_change(5).shift(-5)` at t = `(close[t+5] - close[t]) / close[t]` ✅
- 시점 t의 미래 5일 수익률을 올바르게 계산함
- `compute()` 에서 `label_*` 컬럼은 `dropna()` 제외 처리 — 최근 20일 소실 방지 ✅

---

## 2. 추세 지표 (`_add_trend`)

### 이동평균

```python
ma5, ma20, ma60, ma120 = close.rolling(w).mean()
close_to_ma{w} = close / ma{w} - 1
```

- 5/20/60/120일 단순이동평균 ✅
- `close_to_ma{w}`: 이동평균 대비 현재가 괴리율(소수), 양수 = 이평선 위 ✅

### MACD 🔴 버그

**실제 pandas_ta 컬럼 순서:**
```
iloc[:,0] = MACD_12_26_9   → MACD 선 (EMA12 - EMA26)
iloc[:,1] = MACDh_12_26_9  → 히스토그램 (MACD 선 - Signal)
iloc[:,2] = MACDs_12_26_9  → 시그널 선 (EMA9 of MACD)
```

**현재 코드 매핑:**
```python
df["macd"]        = macd.iloc[:, 0]   # ✅ MACD 선
df["macd_signal"] = macd.iloc[:, 1]   # ❌ 실제로는 히스토그램
df["macd_diff"]   = macd.iloc[:, 2]   # ❌ 실제로는 시그널 선
```

**영향:**
- `macd_diff > 0` 체크 = "시그널 선 > 0" (의도: "히스토그램 > 0" = MACD가 시그널 위)
- 전문가 에이전트의 `'+양전환' if macd_diff > 0` 판단이 엄밀하지 않음
- 히스토그램 기반 크로스오버 감지 불가

**수정 방법:**
```python
df["macd"]        = macd.iloc[:, 0]   # MACD_12_26_9
df["macd_hist"]   = macd.iloc[:, 1]   # MACDh_12_26_9 (히스토그램)
df["macd_signal"] = macd.iloc[:, 2]   # MACDs_12_26_9 (시그널)
df["macd_diff"]   = df["macd"] - df["macd_signal"]  # 명시적 계산
```

### ADX

```python
adx = ta.adx(high, low, close)
# 컬럼 순서: ADX_14, ADXR_14_2, DMP_14, DMN_14
df["adx"] = adx.iloc[:, 0]   # ADX_14
```

- ADX 값만 추출, 25 이상이면 추세 강함 기준 ✅
- DI+/DI- 미사용 (스냅샷 벡터에도 ADX만 포함) — 필요 시 추가 가능

---

## 3. 모멘텀 지표 (`_add_momentum`)

### RSI

```python
rsi = ta.rsi(close, length=14)
```

- 14일 RSI, Wilder smoothing (pandas_ta 기본값) ✅
- 스크리너: growth `45 < rsi < 72`, crisis `rsi < 30` 기준 적절

### Stochastic

```python
stoch = ta.stoch(high, low, close)
# 컬럼 순서: STOCHk_14_3_3, STOCHd_14_3_3, STOCHh_14_3_3
df["stoch_k"] = stoch.iloc[:, 0]   # %K (14일)
df["stoch_d"] = stoch.iloc[:, 1]   # %D (%K의 3일 이동평균)
```

- 표준 14-3-3 설정 ✅
- `STOCHh` (히스토그램 = %K - %D) 미저장 — 필요 시 추가 가능

### CCI

```python
cci = ta.cci(high, low, close)
```

- 기본 14일, `(TP - SMA) / (0.015 × MAD)` ✅
- 범위: 통상 -100 ~ +100 (추세 강도)

### Williams %R

```python
williams_r = ta.willr(high, low, close)
```

- 기본 14일, `(최고가 - 현재가) / (최고가 - 최저가) × -100` ✅
- 범위: -100 ~ 0 (과매도 < -80, 과매수 > -20)

---

## 4. 변동성 지표 (`_add_volatility`)

### Bollinger Bands 🟡 비표준

```python
bb = ta.bbands(close)  # 기본값: length=5, std=2
```

**실제 컬럼 순서:** `BBL_5_2.0_2.0`, `BBM_5_2.0_2.0`, `BBU_5_2.0_2.0`, `BBB_5_2.0_2.0`, `BBP_5_2.0_2.0`

```python
df["bb_lower"] = bb.iloc[:, 0]   # BBL ✅
df["bb_mid"]   = bb.iloc[:, 1]   # BBM ✅
df["bb_upper"] = bb.iloc[:, 2]   # BBU ✅
```

**공식:**
```python
bb_pct   = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)   # BB %B ✅
bb_width = (bb_upper - bb_lower) / (bb_mid + 1e-9)              # BB Width ✅
```

**문제:** pandas_ta 기본값 `length=5` (표준 볼린저 밴드는 20일)
- 5일 BB는 단기 노이즈에 매우 민감
- 위기종목 스크리너에서 `bb_pct < 0.1`을 과매도 기준으로 사용하는데, 5일 기준은 변동성 과대 해석 가능

**권장 수정:**
```python
bb = ta.bbands(close, length=20)   # 표준 20일
```

### ATR

```python
atr = ta.atr(high, low, close)   # 기본 14일
```

- Wilder ATR, 일별 수익률 변동성의 실질적 척도 ✅

### 역사적 변동성

```python
hist_vol_20 = ret_1d.rolling(20).std() * np.sqrt(252)
```

- 20일 일별 수익률 표준편차 × √252 = 연환산 변동성 ✅
- 배당주 스크리너에서 `hist_vol_20 * 100` (%)로 표시

---

## 5. 거래량 지표 (`_add_volume`)

### 거래량 비율

```python
vol_ma20     = volume.rolling(20).mean()
volume_ratio = volume / (vol_ma20 + 1)
```

- `+ 1`은 0 나눗셈 방어용. 거래량 단위가 수십만~수백만 주이므로 실질적 영향 없음 ✅
- (이론적으로는 `+ 1e-9` 또는 `vol_ma20.replace(0, np.nan)` 이 더 명확)

### OBV

```python
obv = ta.obv(close, volume)
```

- 표준 OBV, 가격 방향에 따른 누적 거래량 ✅

### MFI

```python
mfi = ta.mfi(high, low, close, volume)   # 기본 14일
```

- 14일 Money Flow Index, RSI의 거래량 가중 버전 ✅
- 범위: 0~100 (과매도 < 20, 과매수 > 80)

### VWAP 근사치

```python
vwap = amount / (volume + 1)   # amount가 있을 때만
```

- pykrx `amount` = 거래대금(원), `volume` = 거래량(주)
- `amount / volume` = 평균 체결가 = 일별 VWAP 근사치 ✅
- 엄밀한 VWAP(intraday 누적)와 다르지만 일별 분석에서 합리적 대안

---

## 6. 상대강도 지표 (`add_relative_strength`) 🟡

### 상대수익률

```python
market_ret = market_df["close"].pct_change().reindex(df.index)
rel_strength_5d  = ret_5d - market_ret.rolling(5).sum()
rel_strength_20d = ret_20d - market_ret.rolling(20).sum()
```

**문제:** 계산 방식 불일치
- `ret_5d = pct_change(5)` = 복리 5일 수익률: `(P_t / P_{t-5}) - 1`
- `market_ret.rolling(5).sum()` = 단순 합산: `Σ r_t`

복리와 단순 합산은 수익률이 작을 때 근사하지만 엄밀하지 않음.

**권장 수정:**
```python
market_5d  = market_df["close"].pct_change(5).reindex(df.index)
market_20d = market_df["close"].pct_change(20).reindex(df.index)
df["rel_strength_5d"]  = df["ret_5d"] - market_5d
df["rel_strength_20d"] = df["ret_20d"] - market_20d
```

### 베타

```python
beta_20d = ret_1d.rolling(20).cov(market_ret) / market_ret.rolling(20).var()
```

- 수식: β = Cov(R_i, R_m) / Var(R_m) ✅
- **주의:** 20일 롤링은 통계적 노이즈가 큼. 60일 이상 권장
- 배당주 스크리너에서 `beta < 0.8` 방어 기준으로 사용

---

## 7. 52주 위치 (`get_52w_position`)

```python
pct_from_52w_high = close / high_52w - 1   # 음수 (고점 대비 하락률)
pct_from_52w_low  = close / low_52w  - 1   # 양수 (저점 대비 상승률)
window_52w = df.tail(252)                   # 252거래일 ≈ 1년
```

- 공식 및 기간(252일) 모두 표준적 ✅

---

## 8. 재무비율 (`data/collector.py`)

| 지표 | 계산식 | 출처 | 평가 |
|------|--------|------|------|
| ROE (근사) | `EPS / BPS × 100` | pykrx | ✅ 순이익/자기자본의 주당 근사치 |
| ROE (정확) | `순이익 / 자기자본 × 100` | DART | ✅ DART 우선 사용 |
| ROA | `순이익 / 자산총계 × 100` | DART | ✅ |
| 부채비율 | `부채총계 / 자기자본 × 100` | DART | ✅ 한국 기준 (부채/자본) |
| 유동비율 | `유동자산 / 유동부채 × 100` | DART | ✅ |
| 이자보상배율 | `영업이익 / 이자비용` | DART | ✅ |
| 배당성향 | `DPS / EPS × 100` | pykrx | ✅ |
| FCF | `OCF - CAPEX` | DART | ✅ abs(capex) 처리로 부호 안전 |
| Graham Number | `√(22.5 × EPS × BPS)` | 계산 | ✅ 22.5 = PER 15 × PBR 1.5 |
| 섹터대비 PER | `per / sector_median - 1` | 캐시 | ✅ |
| YoY 성장률 | `(curr - prev) / \|prev\| × 100` | DART | ✅ abs(prev)로 음수 기준 기간 안전 처리 |

**Graham Number 전제 조건:** `EPS > 0, BPS > 0` 체크 ✅  
**DART 보고서 시점:** 4월 이후 → 전년도 사업보고서, 1~3월 → 전전년도 ✅

---

## 9. 스크리너 점수 (`agents/screener.py`)

### 성장주 (`_score_growth`)

| 조건 | 배점 | 근거 |
|------|------|------|
| 거래량 비율 / 5 (최대 1) × 2 | 0~2 | 거래량 동반 상승 |
| MACD 양전환 (diff > 0) | 1 | 추세 전환 ← **MACD 버그 영향** |
| RSI 45~72 | 1 | 과열 없는 강세 |
| ret_5d × 20 (최대 2) | 0~2 | 단기 모멘텀 |
| op_income_yoy / 50 (최대 2) | 0~2 | 영업이익 성장 |
| revenue_yoy / 50 (최대 1) | 0~1 | 매출 성장 |

### 가치주 (`_score_value`)

| 조건 | 배점 | 근거 |
|------|------|------|
| PBR 0~1.5: `1.5 - pbr` | 0~1.5 | 순자산 할인 |
| PER 0~15: `(15 - per) / 5` | 0~3 | 저PER 우대 |
| 섹터대비 PER < 0 | 0~1 | 상대 저평가 |
| 섹터대비 PBR < 0 | 0~1 | 상대 저평가 |
| ROE > 5: `roe / 20` (최대 1) | 0~1 | 수익성 |
| 부채비율 < 100 | 1 | 재무 안전 |

### 테마주 (`_score_theme`)

| 조건 | 배점 | 근거 |
|------|------|------|
| volume_ratio > 3/2/1.5 | 1~3 | 거래량 폭발 |
| ret_5d > 5%/2% | 1~2 | 단기 급등 |
| RSI 55~75 | 1 | 모멘텀 과열 전 |
| 공매도비중 > 5% | 0.5 | 숏스퀴즈 잠재력 |

### 배당주 (`_score_dividend`)

| 조건 | 배점 | 근거 |
|------|------|------|
| 배당수익률 > 4%/2% | 1~3 | 고배당 |
| 배당성향 0~70% | 1 | 지속 가능성 |
| FCF > 0 | 2 | 현금흐름 뒷받침 |
| 베타 0~0.8 | 1 | 방어적 특성 |
| 부채비율 < 100 | 1 | 재무 안전 |

### 위기종목 (`_score_crisis`)

| 조건 | 배점 | 근거 |
|------|------|------|
| ret_5d < -10%/-5% | 1~3 | 급락 |
| RSI < 30/40 | 1~3 | 과매도 |
| bb_pct < 0.1/0.2 | 1~2 | BB 하단 ← **5일 BB 비표준 영향** |
| 부채비율 0~100 | 1 | 생존 가능성 |
| 유동비율 > 150% | 1 | 단기 유동성 |

---

## 10. 신호 집계 (`portfolio/aggregator.py`)

### 단일 종목 집계 (`aggregate`)

```python
score_i = SIGNAL_SCORE[signal_i] × confidence_i
weighted_score = mean(score_i) × gate_confidence

BUY  if weighted_score > 0.3
SELL if weighted_score < -0.3
HOLD otherwise
```

- SIGNAL_SCORE: BUY=1.0, HOLD=0.0, SELL=-1.0 ✅
- gate_confidence 곱셈으로 GateNet 불확실성 반영 ✅

### 복합 점수 (`aggregator_node`)

```python
composite = avg_confidence × avg_score + expert_count × 0.5
```

- `avg_confidence × avg_score`: 신뢰도 × 매력도 (0~10) ✅
- `expert_count × 0.5`: 다수 전문가 동의 보너스 ✅
- 최대 이론 점수: 1.0 × 10 + 5 × 0.5 = 12.5

---

## 11. 백테스트 성과지표 (`evaluation/backtest.py`)

| 지표 | 계산식 | 평가 |
|------|--------|------|
| 실제 수익률 | `(exit - entry) / entry` | ✅ |
| 누적 수익률 | `(1 + r).cumprod()` | ✅ |
| ARR (연환산) | `cum[-1]^(252/hold_days/n) - 1` | ✅ |
| MDD | `min((cum - cum.cummax()) / cum.cummax())` | ✅ |
| Sharpe | `mean_r / std_r × √(252/hold_days)` | ✅ 무위험이자율 0 가정 |
| Sortino | `mean_r / downside_std × √(252/hold_days)` | ✅ 하방 편차만 사용 |
| Calmar | `ARR / |MDD|` | ✅ |

**주의:** Sharpe/Sortino는 무위험이자율(rf) = 0 가정. 2026년 기준 한국 기준금리 3~3.5% 수준이므로 실제 위험조정 수익률은 과대평가될 수 있음.

---

## 12. 수정 우선순위

| 우선순위 | 파일 | 라인 | 수정 내용 |
|----------|------|------|-----------|
| 🔴 1 | `feature_engineer.py` | 48~50 | `macd_signal`↔`macd_hist` 이름 수정, `macd_diff` 명시 계산 |
| 🟡 2 | `feature_engineer.py` | 75 | `ta.bbands(close, length=20)` 명시 |
| 🟡 3 | `feature_engineer.py` | 131~132 | `rel_strength` 계산 방식 통일 (`pct_change` 기준) |
| 🟢 4 | `feature_engineer.py` | 133~136 | 베타 윈도우 20 → 60일 고려 |
| 🟢 5 | `evaluation/backtest.py` | 125 | rf 파라미터 추가 (기본값 0 유지) |

---

## 부록: 스냅샷 벡터 피처 목록

`get_snapshot_vector()`가 반환하는 12개 스칼라 피처:

| 피처 | 정상 범위 | 설명 |
|------|-----------|------|
| `returns_series` | -0.3 ~ 0.3 | 최근 20일 일별 수익률 시계열 |
| `ret_5d` | -0.3 ~ 0.3 | 5일 누적 수익률 |
| `ret_20d` | -0.5 ~ 0.5 | 20일 누적 수익률 |
| `rsi` | 0 ~ 100 | RSI (기본값 50) |
| `macd_diff` | 임의 | ⚠️ 실제로는 Signal Line 값 |
| `bb_pct` | 0 ~ 1 | BB 내 위치 (기본값 0.5) |
| `volume_ratio` | 0 ~ 수십 | 거래량/20일평균 (기본값 1) |
| `hist_vol_20` | 0.1 ~ 1.0 | 연환산 역사적 변동성 |
| `close_to_ma20` | -0.3 ~ 0.3 | 20일선 괴리율 |
| `close_to_ma60` | -0.5 ~ 0.5 | 60일선 괴리율 |
| `adx` | 0 ~ 100 | ADX 추세 강도 (기본값 20) |
| `mfi` | 0 ~ 100 | Money Flow Index (기본값 50) |
