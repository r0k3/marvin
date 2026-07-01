import json
import logging

from litellm import completion

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    def __init__(self, model: str = "ollama/qwen3.6:35b-a3b-q4_K_M", api_base: str | None = None):
        self.model = model
        self.api_base = api_base

    def consolidate_episodes(self, episodes: list[str]) -> dict:
        """
        Takes raw episodic markdown logs and extracts permanent semantic facts
        and procedural rules.
        """
        if not episodes:
            return {"semantic": [], "procedural": []}

        prompt = f"""
You are an AI agent's memory consolidation worker (computational sleep).
Analyze the following raw episodic logs from a recent coding session.

Your goal is to extract:
1. "semantic": Permanent architectural facts, decisions, or user preferences.
2. "procedural": Reusable coding rules, steps, or conventions to prevent future errors.
3. "reflective": High-level insights, patterns, or principles realized from the logs.

Output valid JSON ONLY in this exact format:
{{
  "semantic": [
    {{
      "concept": "Database Configuration",
      "predicate": "storage",
      "value": "The project uses PostgreSQL with asyncpg.",
      "aspect": "decision",
      "confidence": 0.8
    }}
  ],
  "procedural": [
    {{"title": "Running Migrations", "rule": "Always run alembic upgrade head."}}
  ],
  "reflective": [
    {{"title": "Database Optimization", "insight": "Run migrations early."}}
  ]
}}

For semantic facts:
- "concept" is the subject/entity the fact is about.
- "predicate" is a short stable property name (for example: "storage",
  "preference", "constraint", "owner", "uses").
- "value" is the fact text.
- "aspect" must be one of: knowledge, preference, decision, goal, problem,
  belief, directive.
- "confidence" is a number from 0.0 to 1.0.

Raw Episodes:
{"\n---\n".join(episodes)}
"""

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.warning("Error during LLM consolidation: %s", e)
            return {"semantic": [], "procedural": []}

    def extract_entity_facts(
        self, entity: str, episodes: list[str], known_facts: list[str] | None = None
    ) -> list[dict]:
        """Phase 1 (episodic -> semantic): extract stable facts about one entity.

        Reviews episodes that mention ``entity`` and returns semantic facts
        about it, skipping anything already in ``known_facts``. Never invents
        facts the episodes do not support.
        """
        if not episodes:
            return []

        known = "\n".join(f"- {fact}" for fact in (known_facts or [])) or "(none)"
        joined = "\n---\n".join(episodes)
        prompt = f"""
You are an AI agent's memory consolidation worker (computational sleep).
Extract stable semantic facts ABOUT "{entity}" from the episodes below.

Rules:
- Only facts about "{entity}", supported by the episodes. Do NOT invent.
- Decompose compound statements into individual atomic facts (one predicate/value each).
- Do NOT repeat any already-known fact.
- "predicate" is a short stable property name (e.g. "uses", "owner", "status").
- "aspect" must be one of: knowledge, preference, decision, goal, problem,
  belief, directive.
- "confidence" is a number from 0.0 to 1.0.

Output valid JSON ONLY:
{{"facts": [{{"predicate": "...", "value": "...", "aspect": "knowledge", "confidence": 0.8}}]}}

Already-known facts about "{entity}" (do not repeat):
{known}

Episodes mentioning "{entity}":
{joined}
"""

        extra: dict[str, object] = {}
        if self.model.startswith("ollama/"):
            extra["think"] = False

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                response_format={"type": "json_object"},
                **extra,
            )
            data = json.loads(response.choices[0].message.content)
            facts = data.get("facts", [])
            return facts if isinstance(facts, list) else []
        except Exception as e:
            logger.warning("Error during entity fact extraction: %s", e)
            return []

    def synthesize_insights(self, aspect: str, facts: list[str]) -> list[dict]:
        """Phase 2 (semantic -> reflective): synthesize cross-fact insights.

        Given facts the agent already knows, all of one ``aspect``, surface
        higher-order patterns, gaps, anomalies, or lessons *across* them. Never
        introduces new facts -- only reorganizes and generalizes what is given.
        Returns a list of ``{title, insight, type, topics}`` dicts.
        """
        if not facts:
            return []

        listed = "\n".join(f"- {fact}" for fact in facts)
        prompt = f"""
You are an AI agent's memory consolidation worker (computational sleep).
Below are facts the agent already knows, all classified as "{aspect}".

Synthesize higher-order REFLECTIVE insights *across* these facts: recurring
patterns, notable gaps, anomalies, or lessons. Do NOT introduce any new facts
or details that are not entailed by the list -- only reorganize and generalize
what is given. If nothing rises above the individual facts, return an empty list.

Output valid JSON ONLY in this exact format:
{{
  "insights": [
    {{
      "title": "Short insight title",
      "insight": "One or two sentences stating the pattern, gap, anomaly, or lesson.",
      "type": "pattern",
      "topics": ["topic-a", "topic-b"]
    }}
  ]
}}

"type" must be one of: pattern, gap, anomaly, lesson.

Facts ({aspect}):
{listed}
"""

        # Disable Qwen3's hidden reasoning on ollama -- it otherwise consumes the
        # token budget and can yield empty output; the native ``think=False``
        # option (unlike the ignored ``/no_think`` prompt switch) disables it.
        extra: dict[str, object] = {}
        if self.model.startswith("ollama/"):
            extra["think"] = False

        try:
            response = completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                api_base=self.api_base,
                response_format={"type": "json_object"},
                **extra,
            )
            data = json.loads(response.choices[0].message.content)
            insights = data.get("insights", [])
            return insights if isinstance(insights, list) else []
        except Exception as e:
            logger.warning("Error during reflective synthesis: %s", e)
            return []
