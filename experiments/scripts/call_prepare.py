import asyncio
from marvin.service import MarvinService
from marvin.config import MarvinSettings

async def main():
    settings = MarvinSettings(embedding_provider="hash")
    service = MarvinService(settings)
    result = service.prepare_session(task="Reviewing Shakespeare Act 1")
    print(result.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
