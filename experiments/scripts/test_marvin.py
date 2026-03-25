import sys
import json
from mcp.server.fastmcp import FastMCP
from marvin.config import MarvinSettings
from marvin.server import create_app

app = create_app(MarvinSettings())

for t in app._tool_manager.list_tools():
    print(t.name)
