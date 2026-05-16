import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent

with open(ROOT / "config" / "settings.yaml", "r", encoding="utf-8") as f:
    SETTINGS = yaml.safe_load(f)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
DART_API_KEY = os.getenv("DART_API_KEY", "")
