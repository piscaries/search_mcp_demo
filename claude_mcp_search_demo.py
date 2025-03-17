#!/usr/bin/env python3
"""
Claude Search MCP Integration

This script demonstrates the integration between Anthropic's Claude and
the Search MCP server, utilizing Claude's function calling capabilities
to search an e-commerce product catalog.

Features:
- Demonstrates Claude's function calling to interact with MCP tools
- Handles tool use blocks from Claude's responses
- Maintains conversation history for coherent multi-turn interactions
- Creates an interactive search experience with real-time product search
"""

import os
import sys
import json
import time
import subprocess
import threading
import select
import re
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv()

# Add the parent directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    print(f"Added {parent_dir} to Python path")

# Define the index name
INDEX_NAME = "ecommerce"

# Initialize the Claude client
claude_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


class MCPClient:
    """Client for interacting with the MCP server."""

    def __init__(self, server_process):
        self.server_process = server_process
        self.message_id = 0
        self.debug_mode = os.environ.get("DEBUG_MODE", "False").lower() == "true"

    def _get_next_id(self):
        """Get the next message ID."""
        self.message_id += 1
        return f"msg-{self.message_id}"

    def list_tools(self):
        """List all available tools from the MCP server."""
        message_id = self._get_next_id()
        message = json.dumps({"id": message_id, "type": "list_tools"}) + "\n"

        if self.debug_mode:
            print("\n🔄 Sending list_tools request to MCP server")
            print(f"📤 Outgoing message: {message.strip()}")

        try:
            self.server_process.stdin.write(message)
            self.server_process.stdin.flush()

            # Wait for response with timeout
            start_time = time.time()
            timeout = 10  # seconds

            while time.time() - start_time < timeout:
                readable, _, _ = select.select(
                    [self.server_process.stdout], [], [], 0.1
                )
                if readable:
                    line = self.server_process.stdout.readline().strip()
                    if line:
                        if self.debug_mode:
                            print(f"📥 Incoming response: {line}")

                        try:
                            response = json.loads(line)
                            if (
                                response.get("id") == message_id
                                and response.get("type") == "list_tools_response"
                            ):
                                if self.debug_mode:
                                    print(
                                        "✅ Successfully received and parsed tools list"
                                    )
                                return response.get("tools", [])
                        except json.JSONDecodeError:
                            print(f"Error parsing response: {line}")
                time.sleep(0.1)

            print("No response received within timeout")
            return []
        except Exception as e:
            print(f"Error listing tools: {e}")
            return []

    def call_tool(self, tool_name, args):
        """Call a tool on the MCP server."""
        message_id = self._get_next_id()
        message = (
            json.dumps(
                {"id": message_id, "type": "tool_call", "tool": tool_name, "args": args}
            )
            + "\n"
        )

        if self.debug_mode:
            print(f"\n🔄 Calling tool: {tool_name}")
            print(f"📤 Arguments: {json.dumps(args, indent=2)}")
            print(f"📤 Outgoing message: {message.strip()}")

        try:
            self.server_process.stdin.write(message)
            self.server_process.stdin.flush()

            # Wait for response with timeout
            start_time = time.time()
            timeout = 15  # seconds

            while time.time() - start_time < timeout:
                readable, _, _ = select.select(
                    [self.server_process.stdout], [], [], 0.1
                )
                if readable:
                    line = self.server_process.stdout.readline().strip()
                    if line:
                        if self.debug_mode:
                            print(f"📥 Incoming response: {line}")

                        try:
                            response = json.loads(line)
                            if (
                                response.get("id") == message_id
                                and response.get("type") == "tool_call_response"
                            ):
                                result = response.get("result", "")

                                # Extract and display the query plan if debug mode is on
                                if (
                                    self.debug_mode
                                    and tool_name == "search"
                                    and "Query plan:" in result
                                ):
                                    match = re.search(
                                        r"Query plan:\s*\n([\s\S]*?)(?=\n\nResults:|\Z)",
                                        result,
                                        re.DOTALL,
                                    )
                                    if match:
                                        try:
                                            plan_text = match.group(1).strip()
                                            print(f"📋 Query Plan: {plan_text}")
                                        except Exception as e:
                                            print(
                                                f"Error displaying query plan: {str(e)}"
                                            )

                                return result
                            elif (
                                response.get("id") == message_id
                                and response.get("type") == "error"
                            ):
                                if self.debug_mode:
                                    print(
                                        f"❌ Error from server: {response.get('error')}"
                                    )
                                return f"Error: {response.get('error')}"
                        except json.JSONDecodeError:
                            print(f"Error parsing response: {line}")
                time.sleep(0.1)

            print("No response received within timeout")
            return "No response received"
        except Exception as e:
            print(f"Error calling tool: {e}")
            return f"Error: {str(e)}"


def start_mcp_server():
    """Start the MCP server in a separate process."""
    print("\n=== Starting MCP Server ===")

    # Start the server process
    process = subprocess.Popen(
        [
            "poetry",
            "run",
            "python",
            os.path.join(current_dir, "run_server.py"),  # Use current directory
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Monitor stderr in a separate thread
    def capture_stderr():
        for line in iter(process.stderr.readline, ""):
            if (
                "OpenAI" in line
                or "Elasticsearch" in line
                or "query plan" in line
                or "ERROR" in line
            ):
                print(f"🖥️ SERVER: {line.strip()}")

    stderr_thread = threading.Thread(target=capture_stderr)
    stderr_thread.daemon = True
    stderr_thread.start()

    # Wait for the ready message
    print("Waiting for server to start...")
    ready_line = process.stdout.readline().strip()

    try:
        ready_json = json.loads(ready_line)
        if ready_json.get("type") == "ready":
            print(f"✅ MCP server started successfully: {ready_json.get('message')}")
            return process
        else:
            print(f"❌ Unexpected server message: {ready_line}")
            process.terminate()
            return None
    except json.JSONDecodeError:
        print(f"❌ Could not parse server message as JSON: {ready_line}")
        process.terminate()
        return None
    except Exception as e:
        print(f"❌ Error starting MCP server: {e}")
        process.terminate()
        return None


def initialize_catalog(client):
    """Initialize the e-commerce catalog by creating test data."""
    print("\n=== Initializing E-commerce Catalog ===")

    # First, delete the existing index if it exists
    print("Cleaning up any existing data...")
    try:
        result = client.call_tool(
            "search", {"query": "DELETE_INDEX", "index": INDEX_NAME}
        )
        print(f"✅ {result}")
    except Exception as e:
        print(f"⚠️ {e}")

    # Create the ecommerce test index with fresh data
    print("Creating test e-commerce catalog...")
    result = client.call_tool("create_ecommerce_test_index", {"index": INDEX_NAME})
    print(f"✅ {result}")

    return "E-commerce catalog initialized successfully with sample products."


def claude_search_conversation(client):
    """
    Run an interactive search conversation with Claude using standard function calling.
    This allows Claude to use MCP tools via its function calling capability.
    """
    print("\n=== Claude E-commerce Search with Tools Integration ===")
    print(
        "Type your product search queries below. Type 'exit', 'quit', or 'bye' to end."
    )

    # Get tools from the MCP server
    tools = client.list_tools()

    # Convert tools to Claude's format
    claude_tools = []
    for tool in tools:
        # For each tool, convert its parameters to the format Claude expects
        parameters = tool.get("parameters", {})
        input_schema = {"type": "object", "properties": {}, "required": []}

        # If parameters is a proper JSON Schema object, extract properties and required fields
        if "properties" in parameters:
            for param_name, param_details in parameters.get("properties", {}).items():
                input_schema["properties"][param_name] = {
                    "type": param_details.get("type", "string"),
                    "description": param_details.get("description", ""),
                }

            # Add required fields if they exist
            if "required" in parameters:
                input_schema["required"] = parameters.get("required", [])

        claude_tools.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "input_schema": input_schema,
            }
        )

    # Set up the system prompt for Claude
    system_prompt = """You are a helpful e-commerce search assistant that can search through product data.
You have access to several search tools:

1. search - For general natural language queries about products
2. search_products_by_category - To find products in a specific category
3. search_products_by_brand - To find products from a specific brand

When a user asks about products, use the appropriate tool to search the catalog.
Present the search results in a clear, helpful way, highlighting key product features.

If no results are found, suggest alternative search terms or approaches.
For follow-up questions, consider the context from previous searches.

Always ensure that you include the 'index' parameter with value 'ecommerce' in your tool calls.
"""

    # Initialize conversation without system message
    messages = []

    # Main conversation loop
    while True:
        # Get user input
        user_query = input("\n📱 User: ")

        # Check for exit command
        if user_query.lower() in ["exit", "quit", "bye"]:
            print(
                "🤖 Claude: Goodbye! Thanks for using the e-commerce search assistant."
            )
            break

        # Skip empty queries
        if not user_query.strip():
            print("🤖 Claude: I didn't catch that. Please try again.")
            continue

        # Add user query to messages
        messages.append({"role": "user", "content": user_query})

        try:
            print("🔍 Processing your request...")

            # Call Claude with tools parameter
            response = claude_client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1024,
                system=system_prompt,  # Use system as a top-level parameter
                messages=messages,
                tools=claude_tools,
            )

            # Extract text content from the response
            message_content = ""
            tool_calls = []

            # Process different content types
            for content in response.content:
                if hasattr(content, "text"):
                    message_content += content.text
                elif hasattr(content, "type") and content.type == "tool_use":
                    # This is a tool use block
                    tool_calls.append(
                        {
                            "name": content.name,
                            "parameters": (
                                content.input if hasattr(content, "input") else {}
                            ),
                        }
                    )
                    message_content += f"[Using tool: {content.name}]"

            # Display Claude's initial response
            print(f"\n🤖 Claude: {message_content}")

            # Add Claude's response to conversation history
            messages.append({"role": "assistant", "content": message_content})

            # Handle tool calls if any
            if tool_calls:
                for tool_call in tool_calls:
                    tool_name = tool_call["name"]
                    tool_params = tool_call["parameters"]

                    print(f"🔧 Claude is using the '{tool_name}' tool")

                    # Ensure index parameter is set for search queries
                    if "index" not in tool_params:
                        tool_params["index"] = INDEX_NAME

                    # Call the MCP tool
                    tool_result = client.call_tool(tool_name, tool_params)

                    # Add tool response to conversation
                    messages.append(
                        {"role": "user", "content": f"Tool result: {tool_result}"}
                    )

                    # Ask Claude to process the tool result
                    final_response = claude_client.messages.create(
                        model="claude-3-opus-20240229",
                        max_tokens=2048,
                        system=system_prompt,  # Use system as a top-level parameter
                        messages=messages,
                    )

                    # Get the final response and safely extract text
                    final_message = ""
                    for content in final_response.content:
                        if hasattr(content, "text"):
                            final_message += content.text

                    messages.append({"role": "assistant", "content": final_message})

                    # Display the formatted response
                    print(f"\n🤖 Claude: {final_message}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("🤖 Claude: I'm having some technical difficulties. Let's try again.")


def main():
    """Run the Claude Search MCP Integration demo."""
    print("=== Claude Search MCP Integration Demo ===")
    print("This demo shows how Claude can use MCP tools for e-commerce search")

    # Check if Anthropic API key is set
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "Error: Anthropic API key not found. Set the ANTHROPIC_API_KEY in your .env file."
        )
        return

    # Start MCP server
    server_process = start_mcp_server()
    if not server_process:
        print("Failed to start MCP server. Exiting demo.")
        return

    try:
        # Create MCP client
        client = MCPClient(server_process)

        # Initialize the catalog
        initialize_catalog(client)

        # Run the interactive search conversation
        print("\n=== Starting Interactive Search Mode ===")
        claude_search_conversation(client)

    finally:
        # Clean up
        print("\n=== Cleaning Up ===")
        print("Terminating MCP server...")
        server_process.terminate()
        try:
            server_process.wait(timeout=5)
            print("✅ MCP server terminated successfully")
        except subprocess.TimeoutExpired:
            print("Server process did not terminate, killing it")
            server_process.kill()
            server_process.wait()

    print("\n=== Demo Completed ===")


if __name__ == "__main__":
    main()
