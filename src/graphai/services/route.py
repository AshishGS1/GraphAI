"""The router agent's Gemini call."""

import logging

from google import genai
from google.genai import types
from pydantic import ValidationError

from graphai.models import RouteDecision

logger = logging.getLogger("graphai.llm")


async def route_prompt(
    gemini: genai.Client, model: str, prompt: str, router_prompt: str
) -> RouteDecision:
    """One tool-less call with structured output."""
    response = await gemini.aio.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=router_prompt,
            response_mime_type="application/json",
            response_json_schema=RouteDecision.model_json_schema(),
        ),
    )
    try:
        return RouteDecision.model_validate_json(response.text or "")
    except ValidationError:
        logger.warning("router output unparseable, marking unroutable: %r", response.text)
        return RouteDecision(route="unroutable", reason="router output unparseable")
