import os

os.environ["LLM_PROVIDER"] = "mock"          # tests never call Gemini
os.environ["LLM_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), ".llm_cache_test")

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:                # runs lifespan: loads models
        yield c
