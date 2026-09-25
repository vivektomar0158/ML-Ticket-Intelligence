from fastapi import FastAPI

from app.config import settings

app = FastAPI(title="Ticket Intelligence ML Service", version="0.1.0")


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


@app.get("/health/ready")
def ready() -> dict:
    # Later phases: report loaded models here.
    return {"status": "ok", "models": [], "gemini_configured": bool(settings.gemini_api_key)}
