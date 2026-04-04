import json
import logging

from litellm import completion

logger = logging.getLogger(__name__)


class ConsolidationEngine:
    def __init__(self, model: str = "ollama/qwen3.5:9b", api_base: str | None = None):
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
    {{"concept": "Database Configuration", "fact": "The project uses PostgreSQL with asyncpg."}}
  ],
  "procedural": [
    {{"title": "Running Migrations", "rule": "Always run alembic upgrade head."}}
  ],
  "reflective": [
    {{"title": "Database Optimization", "insight": "Run migrations early."}}
  ]
}}

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
