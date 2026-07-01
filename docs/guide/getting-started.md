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
        "--vault-path",
        "~/.marvin_vault",
        "--transport",
        "stdio"
      ]
    }
  }
}
```

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

4. Load the **Marvin Skill**. 
   Marvin works best when your agent knows *when* to use it. Copy the contents of [`src/marvin/skill.md`](https://github.com/r0k3/marvin/blob/main/src/marvin/skill.md) into your agent's custom system prompt or instructions field. This teaches the agent the K-Lines philosophy and instructs it to autonomously trigger sleep cycles and log episodes.

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
