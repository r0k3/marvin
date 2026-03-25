import os
import asyncio
import urllib.request
from pathlib import Path
from litellm import completion
from marvin.config import MarvinSettings
from marvin.service import MarvinService
from marvin.git import GitManager
from marvin.extraction import extract_entities, auto_link_markdown

os.environ.setdefault("OPENAI_API_KEY", "your-api-key-here")
os.environ["MARVIN_EXTRACT_MODEL"] = "gpt-5.4"

def download_play():
    url = "https://www.gutenberg.org/cache/epub/1514/pg1514.txt"
    print("Downloading A Midsummer Night's Dream from Project Gutenberg...")
    response = urllib.request.urlopen(url)
    text = response.read().decode('utf-8')
    start_idx = text.find("ACT I.")
    end_idx = text.find("ACT II.")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    return text

async def run_simulation():
    vault_dir = Path("full_play_vault")
    import shutil
    if vault_dir.exists():
        shutil.rmtree(vault_dir)
        
    settings = MarvinSettings(
        vault_path=vault_dir,
        state_dir=vault_dir / ".marvin",
        embedding_provider="hash"
    )
    
    git_manager = GitManager(settings.resolved_vault_path)
    service = MarvinService(settings, git_manager=git_manager)
    
    def trigger_worker_extraction(path_str):
        full_path = settings.resolved_vault_path / path_str
        note = service.vault.read_note(full_path)
        content = note.body
        entities = extract_entities(content)
        if entities:
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

    git_manager.create_worktree("session-full-play")
    play_text = download_play()
    
    chunk_size = 4000
    chunks = [play_text[i:i+chunk_size] for i in range(0, len(play_text), chunk_size)]
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Agent Reading Chunk {i+1} ---")
        agent_prompt = f"""
        You are an AI agent building a memory vault about A Midsummer Night's Dream.
        Read this chunk of the play and output exactly one crucial fact or event from it that you want to remember. 
        Format your response as a simple JSON object: {{"title": "Event Name", "summary": "Short summary", "details": "Longer details"}}
        
        Text:
        {chunk}
        """
        
        response = completion(
            model="gpt-5.4",
            messages=[{"role": "user", "content": agent_prompt}],
            response_format={"type": "json_object"}
        )
        
        import json
        try:
            data = json.loads(response.choices[0].message.content)
            print(f"[Agent Logs Episode]: {data.get('title')}")
            res = service.log_episode(
                title=data.get("title", f"Chunk {i} Notes"),
                summary=data.get("summary", ""),
                details=data.get("details", ""),
                tags=["shakespeare", "full-play"]
            )
            trigger_worker_extraction(res.path)
        except Exception as e:
            print(f"Failed to parse agent output: {e}")
            
    print("\n--- Triggering Custom Computational Sleep ---")
    
    episodes = service.vault.list_notes(kind=service.vault._parse_kind("episodic"))
    texts = [ep.body for ep in episodes]
    
    consolidation_prompt = f"""
    You are an AI agent's memory consolidation worker (computational sleep).
    Analyze the following raw episodic logs.
    Your goal is to extract permanent literary facts about the characters or plot as "semantic" memories.
    Do not extract coding rules.

    Output valid JSON ONLY in this exact format:
    {{
      "semantic": [
        {{"concept": "Character Name or Plot Point", "fact": "The actual fact..."}}
      ]
    }}

    Raw Episodes:
    {"\n---\n".join(texts)}
    """
    
    response = completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": consolidation_prompt}],
        response_format={"type": "json_object"}
    )
    import json
    result = json.loads(response.choices[0].message.content)
    
    print("\n[Consolidation Results]:")
    print(result)
    
    for item in result.get("semantic", []):
        res = service.remember_semantic(concept=item.get("concept", "Fact"), content=item.get("fact", ""))
        trigger_worker_extraction(res.path)

    git_manager.merge_worktree("session-full-play", "master")
    print("\nExperiment Complete. Check the 'full_play_vault' directory.")

if __name__ == "__main__":
    asyncio.run(run_simulation())
