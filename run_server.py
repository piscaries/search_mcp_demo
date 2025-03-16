#!/usr/bin/env python3
"""
Final fixed version of the runner script for the Search MCP server.
This version fixes all the identified issues:
1. Ensures Python path message doesn't interfere with JSON communication
2. Properly handles the ready message
3. Correctly calls the tool functions
"""

import os
import sys
import traceback
import asyncio
import json

# Redirect stdout to stderr temporarily to avoid polluting the JSON communication
original_stdout = sys.stdout
sys.stdout = sys.stderr

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(f"Added {parent_dir} to Python path")

# Set up logging to stderr
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mcp_server_final")

logger.info("Starting final fixed MCP server script")

try:
    logger.info("Importing mcp from search_mcp_pkg.core...")
    from search_mcp_pkg.core import (
        mcp,
        search,
        create_ecommerce_test_index,
        index_product,
        search_products_by_category,
        search_products_by_brand,
        create_test_index,
    )

    logger.info(f"Successfully imported mcp, name: {mcp.name}")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.error(f"Python path: {sys.path}")
    traceback.print_exc()
    sys.exit(1)

# Restore stdout for JSON communication
sys.stdout = original_stdout


# Custom stdio transport with proper ordering
class FixedStdioTransport:
    def __init__(self):
        self.logger = logging.getLogger("mcp_transport")
        self.logger.info("Initializing fixed stdio transport")

    async def read_message(self):
        """Read a message from stdin."""
        self.logger.info("Waiting for input on stdin...")
        try:
            line = sys.stdin.readline().strip()
            if not line:
                self.logger.warning("Empty line received on stdin")
                return None

            self.logger.info(f"Received input: {line}")
            try:
                message = json.loads(line)
                self.logger.info(f"Parsed message: {message}")
                return message
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Error reading from stdin: {e}")
            return None

    async def write_message(self, message):
        """Write a message to stdout."""
        try:
            json_message = json.dumps(message)
            self.logger.info(f"Sending message: {json_message}")
            print(json_message, flush=True)
            self.logger.info("Message sent and flushed")
            return True
        except Exception as e:
            self.logger.error(f"Error writing to stdout: {e}")
            return False


# Map of tool names to actual functions
tool_functions = {
    "search": search,
    "create_ecommerce_test_index": create_ecommerce_test_index,
    "index_product": index_product,
    "search_products_by_category": search_products_by_category,
    "search_products_by_brand": search_products_by_brand,
    "create_test_index": create_test_index,
}


async def run_server():
    """Run the MCP server with the fixed transport."""
    logger.info("Setting up fixed transport")
    transport = FixedStdioTransport()

    # First, send a ready message and ensure it's flushed
    logger.info("Sending ready message")
    await transport.write_message({"type": "ready", "message": "MCP server is ready"})

    # Then enter the main loop
    logger.info("Starting MCP server loop")
    try:
        while True:
            logger.info("Waiting for next message...")
            message = await transport.read_message()

            if message is None:
                logger.warning("Received None message, continuing...")
                continue

            logger.info(f"Processing message: {message}")

            # Handle different message types
            if message.get("type") == "list_tools":
                logger.info("Handling list_tools request")
                try:
                    tools = await mcp.list_tools()
                    logger.info(f"Found {len(tools)} tools")

                    # Convert tools to a serializable format
                    tool_list = []
                    for tool in tools:
                        try:
                            tool_dict = {
                                "name": (
                                    tool.name if hasattr(tool, "name") else str(tool)
                                ),
                                "description": (
                                    tool.description
                                    if hasattr(tool, "description")
                                    else "No description"
                                ),
                            }
                            tool_list.append(tool_dict)
                        except Exception as e:
                            logger.error(f"Error converting tool to dict: {e}")

                    response = {
                        "id": message.get("id", "unknown"),
                        "type": "list_tools_response",
                        "tools": tool_list,
                    }
                    await transport.write_message(response)
                except Exception as e:
                    logger.error(f"Error listing tools: {e}")
                    await transport.write_message(
                        {
                            "id": message.get("id", "unknown"),
                            "type": "error",
                            "error": str(e),
                        }
                    )

            elif message.get("type") == "tool_call":
                logger.info(f"Handling tool call for: {message.get('tool')}")
                try:
                    # Get the tool name and args
                    tool_name = message.get("tool")
                    args = message.get("args", {})

                    # Find the tool function
                    if tool_name in tool_functions:
                        logger.info(f"Found tool function: {tool_name}")

                        # Call the tool function directly
                        logger.info(f"Calling tool function with args: {args}")
                        result = tool_functions[tool_name](**args)
                        logger.info(f"Tool result: {result}")

                        # Send the response
                        response = {
                            "id": message.get("id", "unknown"),
                            "type": "tool_call_response",
                            "result": result,
                        }
                        await transport.write_message(response)
                    else:
                        logger.error(f"Tool not found: {tool_name}")
                        await transport.write_message(
                            {
                                "id": message.get("id", "unknown"),
                                "type": "error",
                                "error": f"Tool not found: {tool_name}",
                            }
                        )

                except Exception as e:
                    logger.error(f"Error calling tool: {e}")
                    traceback.print_exc()
                    await transport.write_message(
                        {
                            "id": message.get("id", "unknown"),
                            "type": "error",
                            "error": str(e),
                        }
                    )

            else:
                logger.warning(f"Unknown message type: {message.get('type')}")
                await transport.write_message(
                    {
                        "id": message.get("id", "unknown"),
                        "type": "error",
                        "error": f"Unknown message type: {message.get('type')}",
                    }
                )

    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception in server loop: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    logger.info("Starting final fixed Search MCP server...")
    try:
        asyncio.run(run_server())
        logger.info("MCP server finished")
    except Exception as e:
        logger.error(f"Error running MCP server: {e}")
        traceback.print_exc()
        sys.exit(1)
