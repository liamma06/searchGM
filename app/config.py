import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY", "")
    elastic_url = os.getenv("ELASTIC_URL", "").strip()
    elastic_api_key = os.getenv("ELASTIC_API_KEY", "").strip()
    elastic_index = os.getenv("ELASTIC_INDEX", "signal_chunks")
    mcp_url = os.getenv("MCP_URL", "http://3.143.20.160/mcp")
    chat_model = os.getenv("CHAT_MODEL", "gpt-4.1")
    fast_model = os.getenv("FAST_MODEL", "gpt-4.1-mini")
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    sentry_dsn = os.getenv("SENTRY_DSN", "").strip()
    sentry_environment = os.getenv("SENTRY_ENVIRONMENT", "hackathon")
    gptzero_api_key = os.getenv("GPTZERO_API_KEY", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    # independent verifier: a different model family than the OpenAI agents, served via OpenRouter
    # reasoning is switched off for it (see verifier.py): with reasoning on it takes ~95 s on real prompts vs ~7 s
    verifier_model = os.getenv("VERIFIER_MODEL", "deepseek/deepseek-v4.1-flash")
    verifier_timeout = float(os.getenv("VERIFIER_TIMEOUT", "20"))
    # identification questions. Measured on the 30 RBC prize questions: lookup, team and auto picked the same company
    # every time, lookup at a 7 s median vs 16 s (team) and a 56 s p90 (auto), so lookup is the default.
    # "team" = one analyst agent per company + cross-model verifier (UI: "Deep team analysis"); "auto" = lookup, escalate if unsure
    identify_flow = os.getenv("IDENTIFY_FLOW", "lookup")
    # sign-in and saved history
    session_secret = os.getenv("SESSION_SECRET", "")
    auth_required = os.getenv("AUTH_REQUIRED", "1") != "0"
    cookie_secure = os.getenv("COOKIE_SECURE", "0") == "1"
    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    mongodb_db = os.getenv("MONGODB_DB", "signal")
    rate_limit_per_hour = int(os.getenv("RATE_LIMIT_PER_HOUR", "60"))
    admin_users = [e.strip().lower() for e in os.getenv("ADMIN_USERS", "").split(",") if e.strip()]
    cache_dir = ROOT / "data" / "cache"


settings = Settings()
