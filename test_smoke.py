"""스모크 테스트: API 호출 없이 핵심 로직 검증"""

import sys
sys.path.insert(0, ".")


def test_text_generator():
    from data.text_generator import generate_pattern_text, generate_news_text

    snap = {
        "ret_5d": 0.043, "ret_20d": 0.091, "rsi": 65.2,
        "macd_diff": 0.003, "bb_pct": 0.72, "volume_ratio": 2.1,
        "hist_vol_20": 0.28, "close_to_ma20": 0.032, "close_to_ma60": 0.058,
        "adx": 28.4, "mfi": 62.3, "returns_series": [0.01] * 20,
    }
    text = generate_pattern_text(
        "005930", "삼성전자", "반도체", "KOSPI", snap, label_5d=0.055
    )
    assert "005930" in text
    assert "반도체" in text
    assert "+4.3%" in text

    news = generate_news_text("005930", "삼성전자", [
        {"rcept_dt": "2025-05-15", "report_nm": "분기보고서", "corp_name": "삼성전자"},
    ])
    assert "분기보고서" in news
    print("  PASS text_generator")
    return text


def test_aggregator():
    from portfolio.aggregator import StockSignal, aggregate, build_portfolio, build_report

    sig = StockSignal(
        ticker="005930", name="삼성전자", sector="반도체", date="2025-05-16",
        gate_result={"experts": ["growth"], "confidence": 0.82, "pattern_type": "성장모멘텀"},
        expert_results=[
            {"expert": "growth", "signal": "BUY", "confidence": 0.78,
             "target_return": 0.07, "horizon_days": 10, "score": 8, "reason": "모멘텀 강함"},
        ],
    )
    sig = aggregate(sig)
    assert sig.final_signal == "BUY"
    assert sig.final_score > 0
    assert sig.final_confidence > 0.5

    # 멀티 전문가: BUY + SELL → HOLD 가능 케이스
    sig2 = StockSignal(
        ticker="000660", name="SK하이닉스", sector="반도체", date="2025-05-16",
        gate_result={"experts": ["growth", "crisis"], "confidence": 0.6, "pattern_type": "혼합"},
        expert_results=[
            {"expert": "growth", "signal": "BUY",  "confidence": 0.55, "target_return": 0.03},
            {"expert": "crisis", "signal": "SELL", "confidence": 0.60, "target_return": -0.04},
        ],
    )
    sig2 = aggregate(sig2)
    assert sig2.final_signal in ("BUY", "HOLD", "SELL")

    # 포트폴리오 빌드
    portfolio = build_portfolio([sig, sig2], top_n=5, min_confidence=0.5)
    assert len(portfolio) >= 1
    assert "ticker" in portfolio.columns

    report = build_report(portfolio, "2025-05-16")
    assert "MERA" in report
    print("  PASS aggregator")


def test_parse_json():
    from agents.base import parse_json_response

    # 정상 JSON
    raw = '{"signal": "BUY", "confidence": 0.75, "target_return": 0.08}'
    result = parse_json_response(raw)
    assert result["signal"] == "BUY"
    assert result["confidence"] == 0.75

    # 마크다운 코드블록 안에 있는 경우
    raw2 = '```json\n{"signal": "HOLD", "confidence": 0.5}\n```'
    result2 = parse_json_response(raw2)
    assert result2["signal"] == "HOLD"

    # 파싱 실패 → 빈 dict
    result3 = parse_json_response("분석 결과: 매수 추천합니다.")
    assert isinstance(result3, dict)
    print("  PASS parse_json_response")


def test_feature_engineer():
    import pandas as pd
    import numpy as np
    from data.feature_engineer import FeatureEngineer

    # 더미 OHLCV 데이터 생성
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    np.random.seed(42)
    close = 70000 + np.cumsum(np.random.randn(n) * 500)
    df = pd.DataFrame({
        "open":   close * 0.99,
        "high":   close * 1.01,
        "low":    close * 0.98,
        "close":  close,
        "volume": np.random.randint(1_000_000, 5_000_000, n).astype(float),
        "amount": (np.random.rand(n) * 250_000_000_000 + 50_000_000_000),
        "changes": np.random.randn(n) * 0.01,
    }, index=dates)

    fe = FeatureEngineer(window=20)
    result = fe.compute(df)

    assert "rsi" in result.columns
    assert "macd_diff" in result.columns
    assert "bb_pct" in result.columns
    assert "volume_ratio" in result.columns
    assert len(result) > 50

    snap = fe.get_snapshot_vector(result, result.index[-1].strftime("%Y-%m-%d"))
    assert snap is not None
    assert "rsi" in snap
    assert len(snap["returns_series"]) == 20
    print("  PASS feature_engineer")


if __name__ == "__main__":
    print("=== 스모크 테스트 시작 ===")
    test_text_generator()
    test_aggregator()
    test_parse_json()
    test_feature_engineer()
    print("\n=== 전체 통과 ===")
