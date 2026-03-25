import os
import re
from pathlib import Path

vault = Path("full_play_vault")

nodes = set()
edges = []

for filepath in vault.rglob("*.md"):
    if ".marvin" in filepath.parts: continue
    
    content = filepath.read_text(encoding="utf-8")
    title_match = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
    
    if not title_match: continue
    title = title_match.group(1).strip()
    
    nodes.add(title)
    
    # find links
    links = re.findall(r"\[\[([^\]]+)\]\]", content)
    for link in links:
        link = link.strip()
        if "[[" in link or "]]" in link: continue 
        nodes.add(link)
        if (title, link) not in edges and title != link:
            edges.append((title, link))

mermaid = ["```mermaid", "graph TD;"]
for node in nodes:
    safe_name = re.sub(r'[^a-zA-Z0-9]', '', node)
    if not safe_name: continue
    mermaid.append(f'  {safe_name}["{node}"]')
    
for src, dst in edges:
    safe_src = re.sub(r'[^a-zA-Z0-9]', '', src)
    safe_dst = re.sub(r'[^a-zA-Z0-9]', '', dst)
    if safe_src and safe_dst:
        mermaid.append(f"  {safe_src} --> {safe_dst}")

mermaid.append("```")

with open("docs/assets/graph.md", "w") as f:
    f.write("\n".join(mermaid))
    
print("Mermaid graph generated at docs/assets/graph.md")
