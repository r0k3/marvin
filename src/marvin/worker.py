import asyncio
import logging
import os

from marvin.broker import MarvinBroker
from marvin.config import MarvinSettings
from marvin.consolidation import ConsolidationEngine
from marvin.extraction import auto_link_markdown, extract_entities
from marvin.git import GitManager
from marvin.service import MarvinService

logger = logging.getLogger(__name__)


async def run_worker():
    logger.info("Starting Marvin Brain Worker...")

    settings = MarvinSettings()
    # NATS URL from environment, or default
    nats_url = os.environ.get("NATS_URL", "nats://127.0.0.1:4222")
    ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

    broker = MarvinBroker(nats_url)
    await broker.connect()

    service = MarvinService(settings)
    git_manager = GitManager(settings.resolved_vault_path)
    consolidation_engine = ConsolidationEngine(api_base=ollama_host)

    async def handle_memory_created(payload: dict):
        path_str = payload.get("path")
        if not path_str:
            return

        logger.info("Worker processing new memory: %s", path_str)
        full_path = settings.resolved_vault_path / path_str
        if not full_path.exists():
            return

        # 1. Read note
        try:
            note = service.vault.read_note(full_path)
            content = note.body
        except Exception as e:
            logger.warning("Could not read note: %s", e)
            return

        # 2. Extract Entities
        entities = extract_entities(content)
        if entities:
            logger.info("Extracted entities: %s", entities)
            # 3. Auto-link
            new_content = auto_link_markdown(content, entities)

            if new_content != content:
                # Update the note with new links
                service.vault.write_note(
                    kind=note.metadata.kind,
                    title=note.metadata.title,
                    body=new_content,
                    tags=note.metadata.tags,
                    links=list(set(note.metadata.links + entities)),
                    existing_path=full_path,
                )

                # Commit via Git
                git_manager.commit(f"chore(graph): auto-linked entities in {note.metadata.title}")

    async def handle_sleep(payload: dict):
        logger.info("Worker starting computational sleep (Consolidation)...")
        # Find all un-consolidated episodic memories
        episodes = service.vault.list_notes(kind=service.vault._parse_kind("episodic"))

        # In a real impl, we'd check for a 'consolidated: true' frontmatter flag.
        # For MVP, we'll just take the 5 most recent.
        recent_eps = sorted(episodes, key=lambda x: x.metadata.created_at, reverse=True)[:5]
        texts = [ep.body for ep in recent_eps]

        if not texts:
            logger.info("No episodes to consolidate.")
            return

        result = consolidation_engine.consolidate_episodes(texts)

        # Write new Semantic facts
        for item in result.get("semantic", []):
            service.remember_semantic(
                concept=item.get("concept", "Learned Fact"),
                content=item.get("fact", ""),
            )

        # Write new Procedural rules
        for item in result.get("procedural", []):
            service.store_procedure(
                title=item.get("title", "New Rule"), steps=[item.get("rule", "")]
            )

        # Write new Reflective insights
        for item in result.get("reflective", []):
            service.reflect(title=item.get("title", "New Insight"), insight=item.get("insight", ""))

        if result.get("semantic") or result.get("procedural"):
            git_manager.commit("chore(sleep): nightly consolidation completed")
            logger.info("Consolidation successful.")

    await broker.subscribe("memory.created", handle_memory_created)
    await broker.subscribe("memory.sleep", handle_sleep)

    logger.info("Worker listening for events...")
    # Keep alive
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(run_worker())
