import sys
sys.path.insert(0, ".")
from agents.base import parse_json_response, _extract_json_object
import json

sample = """{
  "experts": ["growth", "theme"],
  "confidence": 0.85,
  "pattern_type": "단기상승 후 조정 패턴",
  "reason": "최근 수익률 20일 +18.9%이며 MA60 대비 25.67%로 상승 중."
}"""

# 직접 JSON
r = parse_json_response(sample)
print("1. Direct JSON:", r.get("experts"))

# 마크다운 래핑
wrapped = "```json\n" + sample + "\n```"
r2 = parse_json_response(wrapped)
print("2. Wrapped JSON:", r2.get("experts"))

# trailing comma
trailing = '{"signal": "BUY", "score": 7,}'
r3 = parse_json_response(trailing)
print("3. Trailing comma:", r3)

# 실제 Gemini 스타일 (reason에 특수문자)
real = """{
  "signal": "BUY",
  "confidence": 0.75,
  "target_return": 0.035,
  "horizon_days": 7,
  "score": 7,
  "risks": [
    "과매수 구간 진입시 단기 조정 위험",
    "글로벌 매크로 리스크 (금리/환율)"
  ],
  "reason": "RSI 62로 적정 수준이며 MA20 대비 +3.5% 위치."
}"""
r4 = parse_json_response(real)
print("4. Expert JSON:", r4.get("signal"), r4.get("score"))

# json.loads 직접 테스트
try:
    json.loads(sample)
    print("5. json.loads direct: OK")
except Exception as e:
    print("5. json.loads direct FAILED:", e)
