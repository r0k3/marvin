"""Event bus: in-process by default, NATS for the ``[cluster]`` profile.

The vault itself is the durable work queue (``consolidated`` /
``extracted`` frontmatter flags survive crashes and restarts), so events
are only a *nudge* to process it sooner. The default ``InProcessBus``
dispatches handlers as fire-and-forget tasks inside the serve process;
``NatsBus`` publishes over NATS JetStream so a separate Brain Worker can
consume them (multi-process scale-out).
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from ._extras import require

logger = logging.getLogger(__name__)

Handler = Callable[[dict], Awaitable[None]]


class EventBus(Protocol):
    async def connect(self) -> None: ...

    async def publish(self, subject: str, payload: dict) -> None: ...

    async def subscribe(self, subject: str, callback: Handler) -> None: ...

    async def close(self) -> None: ...


class InProcessBus:
    """Single-process bus: handlers run as fire-and-forget asyncio tasks."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}
        self._tasks: set[asyncio.Task] = set()

    async def connect(self) -> None:
        return None

    async def publish(self, subject: str, payload: dict) -> None:
        for handler in self._handlers.get(subject, []):
            task = asyncio.create_task(self._run(handler, subject, payload))
            # Keep a strong reference so tasks are not garbage-collected mid-run.
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _run(self, handler: Handler, subject: str, payload: dict) -> None:
        try:
            await handler(payload)
        except Exception:
            logger.exception("In-process handler for %s failed", subject)

    async def subscribe(self, subject: str, callback: Handler) -> None:
        self._handlers.setdefault(subject, []).append(callback)

    async def close(self) -> None:
        return None


class NatsBus:
    """NATS JetStream bus for the multi-process ``[cluster]`` profile."""

    def __init__(self, nats_url: str = "nats://127.0.0.1:4222"):
        self.nats_url = nats_url
        self.nc: Any = None
        self.js: Any = None

    async def connect(self) -> None:
        try:
            from nats.aio.client import Client as NATS
        except ImportError:
            require("The NATS bus", None, "cluster")
        self.nc = NATS()
        try:
            await self.nc.connect(self.nats_url)
            self.js = self.nc.jetstream()
            # Ensure stream exists
            await self.js.add_stream(name="MARVIN", subjects=["memory.*"])
        except Exception as e:
            logger.warning("Could not connect to NATS (%s). Async events will be disabled.", e)
            self.js = None

    async def publish(self, subject: str, payload: dict) -> None:
        if self.js is None:
            return
        await self.js.publish(subject, json.dumps(payload).encode())

    async def subscribe(self, subject: str, callback: Handler) -> None:
        if self.js is None:
            return

        async def msg_handler(msg):
            data = json.loads(msg.data.decode())
            await callback(data)
            await msg.ack()

        await self.js.subscribe(
            subject, cb=msg_handler, durable=f"worker_{subject.replace('.', '_')}"
        )

    async def close(self) -> None:
        if self.nc is not None:
            await self.nc.close()
