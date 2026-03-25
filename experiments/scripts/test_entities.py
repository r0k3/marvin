from pathlib import Path
from marvin.models import MemoryKind
from marvin.config import MarvinSettings
from marvin.service import MarvinService

settings = MarvinSettings(vault_path=Path("demo_vault"), state_dir=Path("demo_vault/.marvin"), embedding_provider="hash")
service = MarvinService(settings)

# Let's see what notes we have in Semantic
semantic = service.vault.list_notes(kind=MemoryKind("semantic"))
for s in semantic:
    print(f"- {s.title}")
