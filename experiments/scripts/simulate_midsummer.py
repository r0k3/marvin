import os
import asyncio
from pathlib import Path

# Configure API Keys and Models for the test
os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")
os.environ["MARVIN_EXTRACT_MODEL"] = "gpt-5.4"

from marvin.config import MarvinSettings
from marvin.service import MarvinService
from marvin.git import GitManager
from marvin.extraction import extract_entities, auto_link_markdown
from marvin.consolidation import ConsolidationEngine

async def run_simulation():
    vault_dir = Path("demo_vault")
    settings = MarvinSettings(
        vault_path=vault_dir,
        state_dir=vault_dir / ".marvin",
        embedding_provider="hash" # Keep embedding lightweight for test
    )
    
    git_manager = GitManager(settings.resolved_vault_path)
    service = MarvinService(settings, git_manager=git_manager)
    
    # We will simulate the worker logic synchronously for determinism
    def trigger_worker_extraction(path_str):
        print(f"\n[Worker] Extracting entities for {path_str}...")
        full_path = settings.resolved_vault_path / path_str
        note = service.vault.read_note(full_path)
        content = note.body
        
        entities = extract_entities(content)
        if entities:
            print(f"[Worker] Found Entities: {entities}")
            new_content = auto_link_markdown(content, entities)
            if new_content != content:
                service.vault.write_note(
                    kind=note.metadata.kind,
                    title=note.metadata.title,
                    body=new_content,
                    tags=note.metadata.tags,
                    links=list(set(note.metadata.links + entities)),
                    existing_path=full_path
                )
                git_manager.commit(f"chore(graph): auto-linked entities in {note.metadata.title}")
                print(f"[Worker] Graph updated for {note.metadata.title}")

    print(f"1. Vault initialized with git branch: {git_manager.current_branch()}")
    git_manager.create_worktree("session-shakespeare")
    print(f"2. Checked out worktree: {git_manager.current_branch()}")
    
    # --- Phase 1: Conversation Episodes ---
    episodes_data = [
        {
            "title": "Discussing the Fairies",
            "summary": "Talked about Oberon and Titania.",
            "details": "Oberon is the king of the fairies and Titania is the queen. They are fighting over an Indian changeling boy, which is causing chaos in the natural world."
        },
        {
            "title": "Puck's Mistake",
            "summary": "Puck puts the love juice on the wrong Athenian.",
            "details": "Oberon instructs Puck (Robin Goodfellow) to use a magical flower (love-in-idleness) on Demetrius. However, Puck accidentally applies it to Lysander, causing him to fall in love with Helena and abandon Hermia."
        },
        {
            "title": "Bottom's Transformation",
            "summary": "Nick Bottom gets a donkey head.",
            "details": "While the Mechanicals are rehearsing their play 'Pyramus and Thisbe' in the forest, Puck transforms Nick Bottom's head into that of a donkey. Titania, under the love potion's spell, wakes up and falls in love with him."
        }
    ]

    print("--- Starting Conversation Simulation ---")
    for ep in episodes_data:
        print(f"\n[Agent] Logging Episode: {ep['title']}")
        res = service.log_episode(
            title=ep["title"],
            summary=ep["summary"],
            details=ep["details"],
            tags=["shakespeare", "midsummer"]
        )
        # Manually trigger the worker extraction
        trigger_worker_extraction(res.path)

    # --- Phase 2: Consolidation (Sleep) ---
    print("\n--- Triggering Computational Sleep ---")
    engine = ConsolidationEngine(model="gpt-5.4", api_base=None) # Use OpenAI via litellm
    
    # Get un-consolidated episodes
    episodes = service.vault.list_notes(kind=service.vault._parse_kind("episodic"))
    texts = [ep.body for ep in episodes]
    
    print(f"[Worker] Consolidating {len(texts)} episodes...")
    result = engine.consolidate_episodes(texts)
    
    print("\n[Worker] Consolidation Results:")
    print(result)
    
    # Write Semantic facts
    for item in result.get("semantic", []):
        res = service.remember_semantic(
            concept=item.get("concept", "Learned Fact"),
            content=item.get("fact", ""),
            tags=["shakespeare", "lore"]
        )
        trigger_worker_extraction(res.path)
        
    # Write Procedural rules
    for item in result.get("procedural", []):
        res = service.store_procedure(
            title=item.get("title", "New Rule"),
            steps=[item.get("rule", "")],
            tags=["shakespeare", "analysis"]
        )
        trigger_worker_extraction(res.path)
        
    print("\n--- Phase 3: Merging Worktree ---")
    merge_res = git_manager.merge_worktree("session-shakespeare", "master")
    print(f"Merge result: {merge_res}")
        
    print("\nSimulation Complete. Check the 'demo_vault' directory.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
