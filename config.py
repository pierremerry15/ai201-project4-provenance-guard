import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# The course lists meta-llama/llama-4-scout-17b-16e-instruct, which Groq has retired.
# openai/gpt-oss-120b is the course's replacement (Lab 4 starter repo, PR #6).
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

AUDIT_LOG_FILE = os.getenv("AUDIT_LOG_FILE", "logs/audit_log.jsonl")
DB_FILE = os.getenv("DB_FILE", "data/provenance.db")

# --- Input limits ---
MIN_TEXT_CHARS = 20
MAX_TEXT_CHARS = 10_000
MAX_APPEAL_CHARS = 2_000

# --- Rate limits (see planning.md §9 / README "Rate limiting") ---
SUBMIT_RATE_LIMIT = "10 per minute;100 per day"
APPEAL_RATE_LIMIT = "5 per hour"

# --- Scoring (see planning.md §3–4) ---
LLM_WEIGHT = 0.6
STYLO_WEIGHT = 0.4
SINGLE_SIGNAL_SHRINK = 0.6       # LLM unavailable: stylometry only, pulled toward 0.5
SHORT_TEXT_WORDS = 40            # planning.md originally said 60; see "Changes during implementation"
SHORT_TEXT_SHRINK = 0.85

AI_THRESHOLD = 0.80              # ai_score >= this (and gates pass) -> likely_ai
HUMAN_THRESHOLD = 0.35           # ai_score <= this (and gate passes) -> likely_human
AI_GATE_LLM_MIN = 0.65
AI_GATE_STYLO_MIN = 0.55
AI_GATE_MIN_WORDS = 40
HUMAN_GATE_MIN_WORDS = 25
