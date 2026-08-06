"""Brain Worker for the NATS ``[cluster]`` profile.

Consumes ``memory.created`` (entity-extract and auto-link the new note)
and ``memory.sleep`` (full sleep pass) events published by the gateway.
The single-process default does not need it: ``marvin consolidate`` and
the in-process bus cover the same work, and the vault's ``extracted`` /
``consolidated`` flags are the durable queue either way.

Requires the ``cluster`` and ``consolidate`` extras.
"""

import asyncio
import logging

from marvin.bus import NatsBus
from marvin.config import MarvinSettings
from marvin.consolidation import ConsolidationEngine
from marvin.git import GitManager
from marvin.service import MarvinService

logger = logging.getLogger(__name__)


async def run_worker():
    logger.info("Starting Marvin Brain Worker...")

    settings = MarvinSettings()
    bus = NatsBus(settings.nats_url)
    await bus.connect()

    service = MarvinService(settings)
    git_manager = GitManager(settings.resolved_vault_path)
    engine = ConsolidationEngine(model=settings.sleep_model, api_base=settings.sleep_api_base)

    async def handle_memory_created(payload: dict):
        path_str = payload.get("path")
        if not path_str:
            return

        logger.info("Worker processing new memory: %s", path_str)
        full_path = settings.resolved_vault_path / path_str
        if not full_path.exists():
            return

        try:
            note = service.vault.read_note(full_path)
        except Exception as e:
            logger.warning("Could not read note: %s", e)
            return

        if service.extract_note(note):
            git_manager.commit(f"chore(graph): auto-linked entities in {note.metadata.title}")

    async def handle_sleep(payload: dict):
        logger.info("Worker starting computational sleep...")
        report = service.sleep(engine=engine)

        if report.notes_linked or report.facts or report.insights:
            git_manager.commit(
                "chore(sleep): extraction + consolidation "
                "(episodic->semantic, semantic->reflective)"
            )
            logger.info(
                "Sleep: %d notes linked, %d facts, %d insights.",
                report.notes_linked,
                len(report.facts),
                len(report.insights),
            )
        else:
            logger.info("No sleep work produced this pass.")

    await bus.subscribe("memory.created", handle_memory_created)
    await bus.subscribe("memory.sleep", handle_sleep)

    logger.info("Worker listening for events...")
    # Keep alive
    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(run_worker())
