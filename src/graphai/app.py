"""FastAPI app exposing the agent graph."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from google import genai
from google.genai import types

from graphai.config import get_settings
from graphai.router import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.gemini = genai.Client(
        api_key=settings.gemini_api_key.get_secret_value(),
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=3)
        ),
    )
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="graphai", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()