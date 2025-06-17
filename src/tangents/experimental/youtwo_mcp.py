import asyncio
import logging
from pathlib import Path
from pprint import pprint

from mcp import ClientSession, ListToolsResult, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult

from tangents.experimental.utils import BriefEntity, parse_convex_result

MCP_SERVER_PATH = Path("~/Documents/GitHub/YouTwo/").expanduser()
assert MCP_SERVER_PATH.exists(), f"MCP server path does not exist: {MCP_SERVER_PATH}"

server_params = StdioServerParameters(
    command="uv",
    args=[
        "--directory",
        f"{MCP_SERVER_PATH}",
        "run",
        "scripts/run_mcp.py",
    ],
    env=None,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_entities(entities: list[BriefEntity]) -> CallToolResult:
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.call_tool(
            "create_entities",
            {"entities": entities},
        )

async def print_tools() -> ListToolsResult:
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.list_tools()
    

def main():
    res = asyncio.run(create_entities([{'name': 'table', 'entityType': 'furniture'}]))
    res = parse_convex_result(res)
    pprint(res)

if __name__ == "__main__":
    # main()
    res = asyncio.run(print_tools())
    pprint(res)