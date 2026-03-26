# Examples

## Demo Vault

The `demo_vault/` directory contains a sample Marvin vault generated from an agent analyzing Shakespeare's *A Midsummer Night's Dream*. It showcases all four memory types:

| Directory | Memory Type | Purpose |
|-----------|-------------|---------|
| `Semantic/` | Permanent facts | Extracted knowledge about characters, plot, and relationships |
| `Episodic/` | Session logs | Records of specific agent interactions and analyses |
| `Procedural/` | Rules & strategies | Reusable workflows the agent distilled from experience |
| `Reflective/` | Insights | High-level lessons and thematic observations |

Each note is a Markdown file with YAML frontmatter — open the vault directly in [Obsidian](https://obsidian.md) to explore the knowledge graph via `[[wikilinks]]`.
