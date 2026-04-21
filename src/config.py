from dotenv import load_dotenv
import os
import re
import time
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, (
    "GROQ_API_KEY missing from .env — add it with: "
    "echo 'GROQ_API_KEY=gsk_...' >> .env"
)

_MODEL_CANDIDATES = [
    os.getenv("GROQ_MODEL", "").strip(),
    "openai/gpt-oss-120b",
    "groq/compound",
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
]
_MODEL_CANDIDATES = [m for m in _MODEL_CANDIDATES if m]

def _try_build_llm(model_name: str) -> ChatGroq:
    return ChatGroq(model=model_name, api_key=GROQ_API_KEY, temperature=0.1)

llm = None
for _candidate in _MODEL_CANDIDATES:
    try:
        _test_llm = _try_build_llm(_candidate)
        from langchain_core.messages import HumanMessage
        _test_llm.invoke([HumanMessage(content="Reply OK")])
        llm = _test_llm
        print(f"config.py: LLM ready (model={_candidate})")
        break
    except Exception:
        continue

if llm is None:
    llm = _try_build_llm(_MODEL_CANDIDATES[0])
    print(f"config.py: LLM created without probe (model={_MODEL_CANDIDATES[0]})")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
print("config.py: Embedder (all-MiniLM-L6-v2) ready")

def _is_rate_limit_error(error: Exception) -> bool:
    text = str(error).lower()
    return any(
        token in text
        for token in [
            "rate limit",
            "rate_limit_exceeded",
            "error code: 429",
            "status code: 429",
            "tokens per day",
        ]
    )

def _retry_after_seconds(error: Exception, default_seconds: float = 2.0) -> float:
    text = str(error).lower()
    minute_match = re.search(r"try again in\s+(\d+)m([0-9.]+)s", text)
    if minute_match:
        minutes = float(minute_match.group(1))
        seconds = float(minute_match.group(2))
        return min(300.0, minutes * 60.0 + seconds + 0.5)

    second_match = re.search(r"try again in\s+([0-9.]+)s", text)
    if second_match:
        return min(300.0, float(second_match.group(1)) + 0.5)

    return default_seconds

def llm_invoke(messages, max_retries: int = 2):
    """Invoke LLM with lightweight 429 retry handling."""
    attempts = 0
    while True:
        try:
            return llm.invoke(messages)
        except Exception as error:
            if _is_rate_limit_error(error) and attempts < max_retries:
                wait_seconds = _retry_after_seconds(error, default_seconds=2.0 * (attempts + 1))
                attempts += 1
                time.sleep(wait_seconds)
                continue
            raise
