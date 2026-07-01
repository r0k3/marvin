from __future__ import annotations

import logging
import os
import re

try:
    import langextract as lx
    from langextract.core import data

    LANGEXTRACT_AVAILABLE = True
except ImportError:
    LANGEXTRACT_AVAILABLE = False

logger = logging.getLogger(__name__)


def _fallback_extract(text: str) -> list[str]:
    """Fallback regex for capitalized multi-word concepts if langextract fails or is unavailable."""
    return list(set(re.findall(r"\b[A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*\b", text)))


def extract_entities(text: str) -> list[str]:
    """
    Uses Google's langextract to extract named entities (Organizations, Products, Concepts).
    It leverages the LLM configured in the environment to perform zero-shot extraction.
    """
    if not LANGEXTRACT_AVAILABLE:
        return _fallback_extract(text)

    # We provide a few-shot example to teach the schema to langextract
    examples = [
        data.ExampleData(
            text="The Marvin server connects to the NATS broker. It uses Postgres for storage.",
            extractions=[
                data.Extraction("Entity", "Marvin server"),
                data.Extraction("Entity", "NATS broker"),
                data.Extraction("Entity", "Postgres"),
            ],
        )
    ]

    # Use OLLAMA as default if no external API key is provided, matching our V2 Docker setup.
    # User can override via standard LANGEXTRACT_API_KEY / GEMINI_API_KEY environment variables
    # or by setting MARVIN_EXTRACT_MODEL (e.g. 'gpt-5.4' or 'ollama/qwen3.6:35b-a3b-q4_K_M')
    model_id = os.environ.get("MARVIN_EXTRACT_MODEL")
    if not model_id:
        has_api_key = any(
            k in os.environ for k in ("LANGEXTRACT_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
        )
        model_id = "gpt-5.4" if has_api_key else "ollama/qwen3.6:35b-a3b-q4_K_M"

    try:
        doc = lx.extract(
            text_or_documents=text,
            prompt_description=(
                "Extract the names of key software components, systems,"
                " concepts, or entities mentioned in the text."
            ),
            examples=examples,
            model_id=model_id,
            # We suppress parse errors so a bad LLM output doesn't crash the worker
            resolver_params={"suppress_parse_errors": True},
        )

        entities = []
        # If it's a single document, extractions is a list of Extraction objects
        for ext in getattr(doc, "extractions", []):
            is_entity = getattr(ext, "extraction_class", None) == "Entity"
            has_text = isinstance(getattr(ext, "extraction_text", None), str)
            if is_entity and has_text:
                entities.append(ext.extraction_text.strip())

        # Deduplicate and clean
        final_entities = list(set([e for e in entities if len(e) > 2]))
        if not final_entities:
            return _fallback_extract(text)

        return final_entities

    except Exception as e:
        logger.warning("LangExtract LLM extraction failed: %s. Using fallback regex.", e)
        return _fallback_extract(text)


def auto_link_markdown(content: str, known_entities: list[str]) -> str:
    """Injects [[wikilinks]] into text for known entities."""
    linked_content = content
    for entity in known_entities:
        # Don't double link
        if f"[[{entity}]]" in linked_content:
            continue

        # Replace occurrences (case preserving, word boundary)
        pattern = re.compile(rf"\b({re.escape(entity)})\b", re.IGNORECASE)
        # We use a lambda to preserve original case of the text but wrap in wikilinks
        linked_content = pattern.sub(r"[[\1]]", linked_content)

    return linked_content
