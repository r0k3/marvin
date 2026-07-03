# Getting Started

## Quick Installation

For standard usage without the advanced worker node, install the core package
straight from GitHub:

```bash
uv tool install git+https://github.com/r0k3/marvin
```

### Starting the Local MCP Gateway

If your AI agent supports configuring MCP servers via standard I/O streams:

```json
{
  "mcpServers": {
    "marvin": {
      "command": "marvin",
      "args": [
        "serve",
        "--vault-path",
        "~/.marvin_vault",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

Installing also gives you the [AXI command line](../reference/cli.md): run
`marvin` with no arguments for a live vault dashboard, `marvin search <query>`
for token-efficient recall, and `marvin --help` for the full command list.

## Running the Advanced Cluster (Docker)

To utilize the **Background Brain Worker** (for automatic consolidation and deep knowledge graph extraction via Google's `langextract`):

1. Clone the repository:
   ```bash
   git clone https://github.com/r0k3/marvin.git
   cd marvin
   ```

2. Start the cluster:
   ```bash
   docker compose up -d
   ```

3. Download the local consolidation model (only required on first boot — any
   litellm-supported model works; this is the default):
   ```bash
   docker exec -it marvin-ollama-1 ollama pull qwen3.6:35b-a3b-q4_K_M
   ```

4. Install the **Marvin Skill**.
   Marvin works best when your agent knows *when* to use it. The bundled
   `marvin-memory` skill teaches exactly that — storage signals per memory
   type, recall-before-answering, the template feedback loop, and session
   lifecycle:
   ```bash
   marvin skill install          # Claude Code, project-level (.claude/skills/)
   marvin skill install --user   # user-level (~/.claude/skills/)
   marvin skill show             # print it, to paste into any other harness
   ```
   See the [Agent Skills guide](skills.md) for what it teaches and why.

## Configuring Your Agent (MCP Clients)

Marvin communicates via the Model Context Protocol (MCP). Here is how to configure the most popular agentic harnesses to connect to the Dockerized Marvin cluster (which runs on `http://localhost:8421/sse` by default).

### Goose
Add the following to your `~/.config/goose/config.yaml`:

```yaml
extensions:
  marvin:
    enabled: true
    type: sse
    name: marvin
    uri: http://localhost:8421/sse
```

### Claude Desktop
Add the following to your Claude configuration file (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "marvin": {
      "command": "curl",
      "args": ["-s", "http://localhost:8421/sse"],
      "env": {}
    }
  }
}
```
*(Note: Claude desktop currently prefers stdio for local processes. If SSE via curl behaves inconsistently in your environment, use the `stdio` command method shown in the Quick Installation above).*

### Cursor
In Cursor, go to **Settings > Features > MCP Servers** and add a new server:
1. Click **Add New MCP Server**
2. **Type:** `sse`
3. **URL:** `http://localhost:8421/sse`

### OpenCode
For OpenCode CLI agents, provide the server via the configuration block or CLI flags depending on your version:

```json
{
  "mcp_servers": {
    "marvin": {
      "transport": "sse",
      "endpoint": "http://localhost:8421/sse"
    }
  }
}
```

### Gemini
If you are using a Gemini-powered agent loop that supports MCP:

```json
{
  "mcp": {
    "endpoints": [
      {
        "name": "marvin",
        "url": "http://localhost:8421/sse"
      }
    ]
  }
}
```
