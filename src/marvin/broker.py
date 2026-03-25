import json
from typing import Callable, Awaitable
from nats.aio.client import Client as NATS
from nats.js.client import JetStreamContext


class MarvinBroker:
    def __init__(self, nats_url: str = "nats://127.0.0.1:4222"):
        self.nats_url = nats_url
        self.nc = NATS()
        self.js: JetStreamContext | None = None

    async def connect(self):
        try:
            await self.nc.connect(self.nats_url)
            self.js = self.nc.jetstream()
            # Ensure stream exists
            await self.js.add_stream(name="MARVIN", subjects=["memory.*"])
        except Exception as e:
            print(
                f"Warning: Could not connect to NATS ({e}). Async events will be disabled."
            )
            self.js = None

    async def publish(self, subject: str, payload: dict):
        if self.js is None:
            return
        await self.js.publish(subject, json.dumps(payload).encode())

    async def subscribe(
        self, subject: str, callback: Callable[[dict], Awaitable[None]]
    ):
        if self.js is None:
            return

        async def msg_handler(msg):
            data = json.loads(msg.data.decode())
            await callback(data)
            await msg.ack()

        await self.js.subscribe(
            subject, cb=msg_handler, durable=f"worker_{subject.replace('.', '_')}"
        )

    async def close(self):
        await self.nc.close()
