"""Gemini 실제 응답 포맷 확인 및 JSON 파싱 테스트"""
import sys
sys.path.insert(0, ".")
from agents.base import call_llm, parse_json_response
from config import SETTINGS

print("max_tokens:", SETTINGS["llm"]["max_tokens"])

system = "당신은 주식 분석 전문가입니다."
user = """다음 JSON 형식으로만 응답하세요 (다른 텍스트 없이):
{
  "signal": "BUY" 또는 "HOLD" 또는 "SELL",
  "confidence": 0.0~1.0,
  "score": 1~10,
  "reason": "한 문장 이유"
}

삼성전자(005930) RSI=62, MA20대비+3%, 20일수익률+5% 분석 결과는?"""

print("Gemini 호출 중...")
raw = call_llm(system, user)
print("RAW 응답 (앞 500자):\n", raw[:500])
print("\n파싱 결과:", parse_json_response(raw))
