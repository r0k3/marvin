import asyncio
from marvin.broker import MarvinBroker

async def main():
    broker = MarvinBroker()
    await broker.connect()
    print("Connected to broker.")
    await broker.publish("memory.sleep", {"trigger": "agent"})
    print("Consolidation requested. The brain worker is now processing.")
    await broker.close()

if __name__ == "__main__":
    asyncio.run(main())
