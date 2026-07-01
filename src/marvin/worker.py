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
        # Phase 1 (episodic -> semantic): entity-scoped extraction with a
        # per-entity threshold; consumed episodes are marked consolidated.
        facts = service.consolidate_semantic(engine=consolidation_engine)

        # Phase 2 (semantic -> reflective): synthesize insights from the
        # accumulated semantic facts, grounded in what is stored.
        reflections = service.consolidate_reflective(engine=consolidation_engine)

        if facts or reflections:
            git_manager.commit(
                "chore(sleep): consolidation (episodic->semantic, semantic->reflective)"
            )
            logger.info("Consolidation: %d facts, %d insights.", len(facts), len(reflections))
        else:
            logger.info("No consolidation produced this pass.")

    await broker.subscribe("memory.created", handle_memory_created)
    await broker.subscribe("memory.sleep", handle_sleep)

    logger.info("Worker listening for events...")
    # Keep alive
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(run_worker())
