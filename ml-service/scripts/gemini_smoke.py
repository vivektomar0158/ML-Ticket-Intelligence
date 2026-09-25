"""Phase 0 smoke test: verify GEMINI_API_KEY works and pick a model.

Run:  uv run python scripts/gemini_smoke.py
"""
import sys
import time

from google import genai

from app.config import settings

if not settings.gemini_api_key:
    sys.exit("GEMINI_API_KEY is empty. Put it in .env at the repo root.")

client = genai.Client(api_key=settings.gemini_api_key)

models = [m.name for m in client.models.list() if "generateContent" in (m.supported_actions or [])]
print(f"{len(models)} generateContent models; flash-tier:")
for name in models:
    if "flash" in name:
        print("  ", name)

print(f"\nUsing GEMINI_MODEL={settings.gemini_model}")
t = time.perf_counter()
resp = client.models.generate_content(
    model=settings.gemini_model,
    contents='Classify this ticket as one of BILLING, LOGIN_ACCESS, BUG. Reply JSON {"category": ...}. '
             "Ticket: Can't log in since your update, getting error 403.",
    config={"response_mime_type": "application/json", "temperature": 0},
)
print(f"latency={time.perf_counter() - t:.2f}s  response={resp.text}")
print("usage:", resp.usage_metadata)
