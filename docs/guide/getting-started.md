# Getting Started

## Quick Installation

Marvin runs as a **single process** by default — vault, hybrid index, MCP
server, and CLI in one install, with no broker or background daemon. Install
the core package straight from GitHub:

```bash
uv tool install git+https://github.com/r0k3/marvin
```

The base install is retrieval-only (zero LLM dependencies). Two optional
extras unlock more:

```bash
# LLM "computational sleep": entity extraction + two-phase consolidation
uv tool install 'marvin-memory[consolidate] @ git+https://github.com/r0k3/marvin'

# run `marvin consolidate` any time to drain the queue — the vault's
# `extracted` / `consolidated` flags track exactly what is pending
```

At any point, `marvin doctor` reports the install state of every component
(embedder, GPU, extras, sleep endpoint, skill) with the exact command that
fixes anything missing.

### Auto-recall hooks (memory without asking)

By default, memory only works when the agent chooses to call a tool. The
auto-recall hooks remove that dependency — the host runs a fast `marvin`
command at session start and on each user prompt, and its stdout is
injected into the agent's context:

```bash
marvin hooks install --host claude          # project (.claude/settings.json)
marvin hooks install --host claude --user   # user-level (~/.claude/settings.json)
```

What gets injected (budgeted, ~2000 / ~1000 chars, tunable via
`MARVIN_HOOK_SESSION_BUDGET_CHARS` / `MARVIN_HOOK_PROMPT_BUDGET_CHARS`):

- **Session start** — your highest-confidence facts (current project's
  first, via the auto `project/<owner>-<repo>` tag) and recent episodes.
  Embedder-free, so it costs ~0.4 s.
- **Each user prompt** — lexical recall (FTS + entity graph; the dense
  stream is skipped so no model ever loads on this path) against the
  prompt text, plus a nudge when the prompt matches a **correction cue**
  ("no, actually…", "we no longer use…") so the agent persists the fix
  with `marvin remember ... --reason "user correction"`.

Hooks always exit 0 — a broken vault never breaks a session. Dry-run them
any time: `marvin hook session-start`, or
`echo '{"prompt":"..."}' | marvin hook user-prompt`. For other hosts,
`marvin hooks show --host <h>` prints what to wire manually.

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

## Running the Advanced Cluster (Docker, optional)

The single-process default covers most deployments: the serve process runs
sleep passes in-process, and `marvin consolidate` works on demand. For an
**always-on** setup where a dedicated **Brain Worker** processes events
continuously (NATS-brokered, `MARVIN_BUS=nats`), run the Docker cluster:

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
