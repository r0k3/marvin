from __future__ import annotations

import logging
import os
import re
from importlib.util import find_spec

logger = logging.getLogger(__name__)


def langextract_available() -> bool:
    """Cheap installed-check; the actual (slow) langextract import is deferred
    to first use so the CLI never pays it on the read/write path."""
    return find_spec("langextract") is not None


def _resolve_extract_model() -> str:
    """Model id for langextract's zero-shot extraction.

    Local Ollama is the default unless an external API key is present.
    Override via MARVIN_EXTRACT_MODEL (e.g. 'gpt-5.4' or a local model).
    langextract's provider registry matches bare model names (^qwen,
    ^llama, ...), so a litellm-style 'ollama/' prefix breaks resolution —
    found on the first real sleep pass — and is stripped here to keep both
    config styles working.
    """
    model_id = os.environ.get("MARVIN_EXTRACT_MODEL")
    if not model_id:
        has_api_key = any(
            k in os.environ for k in ("LANGEXTRACT_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY")
        )
        model_id = "gpt-5.4" if has_api_key else "qwen3.6:35b-a3b-q4_K_M"
    return model_id.removeprefix("ollama/")


def _fallback_extract(text: str) -> list[str]:
    """Fallback regex for capitalized multi-word concepts if langextract fails or is unavailable."""
    return list(set(re.findall(r"\b[A-Z][a-zA-Z]*(?: [A-Z][a-zA-Z]*)*\b", text)))


def extract_entities(text: str) -> list[str]:
    """
    Uses Google's langextract to extract named entities (Organizations, Products, Concepts).
    It leverages the LLM configured in the environment to perform zero-shot extraction.
    """
    if not langextract_available():
        return _fallback_extract(text)

    import langextract as lx
    from langextract.core import data

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

    model_id = _resolve_extract_model()

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
