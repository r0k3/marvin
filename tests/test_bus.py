"""In-process event bus: dispatch, handler-failure isolation, no-op default."""

from __future__ import annotations

import asyncio

from marvin.bus import InProcessBus


def test_publish_dispatches_to_subscribers():
    async def scenario():
        bus = InProcessBus()
        got: list[dict] = []

        async def handler(payload: dict) -> None:
            got.append(payload)

        await bus.subscribe("memory.sleep", handler)
        await bus.publish("memory.sleep", {"trigger": "test"})
        await asyncio.gather(*bus._tasks)
        return got

    assert asyncio.run(scenario()) == [{"trigger": "test"}]


def test_publish_without_subscribers_is_noop():
    async def scenario():
        bus = InProcessBus()
        await bus.publish("memory.created", {"path": "x.md"})

    asyncio.run(scenario())


def test_handler_failure_does_not_propagate_or_block_others():
    async def scenario():
        bus = InProcessBus()
        ran: list[str] = []

        async def bad(payload: dict) -> None:
            raise ValueError("boom")

        async def good(payload: dict) -> None:
            ran.append("good")

        await bus.subscribe("s", bad)
        await bus.subscribe("s", good)
        await bus.publish("s", {})
        await asyncio.gather(*bus._tasks)
        return ran

    assert asyncio.run(scenario()) == ["good"]
