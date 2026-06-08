"""프로젝트 전역 설정 로드

환경변수(.env)와 YAML 설정 파일(config/settings.yaml)을 읽어
프로젝트 전반에서 사용하는 상수를 정의한다.

.env 파일 필수 항목:
  ANTHROPIC_API_KEY  - Claude LLM API 키
  OPENAI_API_KEY     - 임베딩 API 키 (text-embedding-3-small)
  KRX_ID / KRX_PW   - KRX 로그인 (pykrx 사용 시, 없어도 FDR fallback 동작)
  SUPABASE_URL/KEY   - 포트폴리오 DB 저장용 (선택)
  DART_API_KEY       - DART 공시 조회용 (선택)
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (프로젝트 루트의 .env)
load_dotenv()

# 프로젝트 루트 경로 (config/ 폴더의 상위)
ROOT = Path(__file__).parent.parent

# settings.yaml 로드: 모델명·파라미터·경로·파이프라인 설정 전부 포함
with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)

# API 키 및 외부 서비스 URL (환경변수에서 읽음, 없으면 빈 문자열)
DART_API_KEY = os.getenv("DART_API_KEY", "")         # DART 공시 (선택)
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")  # Supabase REST URL
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")         # Supabase anon key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")     # OpenAI 임베딩 (필수)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")  # Claude LLM (필수)
