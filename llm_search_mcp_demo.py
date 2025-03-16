#!/usr/bin/env python3
"""
Enhanced demo script showing how to integrate the Search MCP server with a language model.
This script demonstrates all steps of the process:
1. Starting the MCP server
2. Connecting to the server with a client
3. User sending a query
4. LLM processing the query
5. MCP server generating a query plan with OpenAI
6. MCP server executing the search against Elasticsearch
7. LLM formatting and presenting the results to the user
"""

import os
import sys
import json
import time
import subprocess
import threading
import select
from dotenv import load_dotenv
import re

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


class MCPClient:
    """Simple client for interacting with the MCP server."""

    def __init__(self, server_process):
        self.server_process = server_process
        self.message_id = 0
        self.debug_mode = True  # Enable debug mode to see detailed steps

    def _get_next_id(self):
        """Get the next message ID."""
        self.message_id += 1
        return f"msg-{self.message_id}"

    def list_tools(self):
        """List all available tools from the MCP server."""
        message_id = self._get_next_id()
        message = json.dumps({"id": message_id, "type": "list_tools"}) + "\n"

        if self.debug_mode:
            print("\n🔄 STEP: Sending list_tools request to MCP server")
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
        """Call a tool on the MCP server with detailed step logging."""
        message_id = self._get_next_id()
        message = (
            json.dumps(
                {"id": message_id, "type": "tool_call", "tool": tool_name, "args": args}
            )
            + "\n"
        )

        if self.debug_mode:
            print(f"\n🔄 STEP 1: LLM decides to use the '{tool_name}' tool")
            print(
                f"🔄 STEP 2: LLM prepares tool call with arguments: {json.dumps(args, indent=2)}"
            )
            print(f"🔄 STEP 3: Sending message to MCP server")
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
                            print(f"🔄 STEP 6: Receiving response from MCP server")
                            print(f"📥 Incoming response: {line}")

                        try:
                            response = json.loads(line)
                            if (
                                response.get("id") == message_id
                                and response.get("type") == "tool_call_response"
                            ):
                                result = response.get("result", "")

                                # Extract and display the query plan
                                if tool_name == "search" and "Query plan:" in result:
                                    if self.debug_mode:
                                        print(
                                            "\n🔄 STEP 4: OpenAI generated a query plan"
                                        )
                                        # Extract the query plan JSON
                                        match = re.search(
                                            r"Query plan: (\{.*\})", result, re.DOTALL
                                        )
                                        if match:
                                            try:
                                                query_plan = json.loads(match.group(1))
                                                print(
                                                    f"📋 Query Plan: {json.dumps(query_plan, indent=2)}"
                                                )
                                                print(
                                                    "\n🔄 STEP 5: Elasticsearch executed the search based on the query plan"
                                                )
                                            except json.JSONDecodeError:
                                                print(
                                                    f"Could not parse query plan: {match.group(1)}"
                                                )

                                if self.debug_mode:
                                    print(
                                        "🔄 STEP 7: Client parses the response and returns it to the LLM"
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
        ["poetry", "run", "python", "run_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Monitor stderr in a separate thread
    def capture_stderr():
        for line in iter(process.stderr.readline, ""):
            # Only print server logs if they contain interesting information
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


def format_search_results(query, result):
    """
    Simulate how an LLM would format search results in a more natural way.
    This function demonstrates STEP 8: LLM formats and presents the results to the user.
    """
    print("\n🔄 STEP 8: LLM formats and presents the results to the user")

    # Extract query plan if available
    query_plan_explanation = ""
    if "Query plan:" in result:
        match = re.search(r'"explanation": "(.*?)"', result, re.DOTALL)
        if match:
            query_plan_explanation = match.group(1)

    # Check if products were found
    if "No products found" in result:
        response = f"I searched for '{query}' but couldn't find any matching products. "
        if query_plan_explanation:
            response += (
                f"\n\nI approached this search by {query_plan_explanation.lower()}"
            )
        return response

    # Extract product information if available
    products = []
    product_sections = re.findall(
        r"Product \d+:(.*?)(?=Product \d+:|$)", result, re.DOTALL
    )

    for section in product_sections:
        product = {}
        name_match = re.search(r"Name: (.*?)(?:\n|$)", section)
        if name_match:
            product["name"] = name_match.group(1).strip()

        price_match = re.search(r"Price: \$([\d.]+)", section)
        if price_match:
            product["price"] = float(price_match.group(1))

        brand_match = re.search(r"Brand: (.*?)(?:\n|$)", section)
        if brand_match:
            product["brand"] = brand_match.group(1).strip()

        rating_match = re.search(r"Rating: ([\d.]+)/5", section)
        if rating_match:
            product["rating"] = float(rating_match.group(1))

        description_match = re.search(r"Description: (.*?)(?:\n|$)", section)
        if description_match:
            product["description"] = description_match.group(1).strip()

        if product:
            products.append(product)

    # Format a natural language response
    if products:
        response = f"Based on your search for '{query}', I found {len(products)} relevant products:\n\n"

        # Add separator line for clarity
        response += "-" * 60 + "\n"

        for i, product in enumerate(products, 1):
            response += f"{i}. {product.get('name', 'Unknown Product')}\n"
            response += f"   Brand: {product.get('brand', 'Unknown Brand')}\n"
            response += f"   Price: ${product.get('price', 0):.2f}\n"
            response += f"   Rating: {product.get('rating', 0)}/5\n"
            if "description" in product:
                response += f"   {product['description']}\n"
            response += "-" * 60 + "\n"

        if query_plan_explanation:
            response += (
                f"\nI approached this search by {query_plan_explanation.lower()}"
            )

        return response
    else:
        return f"I searched for '{query}' but couldn't extract product details from the results."


def simulate_enhanced_llm_conversation(client):
    """Simulate a conversation with an LLM using the MCP tools with detailed steps."""
    print("\n=== Enhanced LLM Conversation Simulation ===")

    # First, delete the existing index if it exists
    print("\n🤖 LLM: Let me first clean up any existing data...")
    try:
        # We'll use a direct Elasticsearch call through the search tool with a special command
        result = client.call_tool(
            "search", {"query": "DELETE_INDEX", "index": INDEX_NAME}
        )
        print(f"🤖 LLM: {result}")
    except Exception as e:
        print(
            f"🤖 LLM: Note: Could not delete existing index. This is fine if it doesn't exist yet. Error: {e}"
        )

    # Create the ecommerce test index with fresh data
    print("\n🤖 LLM: Now I'll create a fresh set of product data to search...")
    result = client.call_tool("create_ecommerce_test_index", {"index": INDEX_NAME})
    print(f"🤖 LLM: {result}")

    # Get available tools
    print("\n📱 User: What tools do you have available?")
    print("🤖 LLM: Let me check what tools I have available...")
    tools = client.list_tools()

    if not tools:
        print("🤖 LLM: I don't have any tools available.")
        return

    print(
        f"🤖 LLM: I have {len(tools)} tools available that can help with e-commerce searches:"
    )
    for i, tool in enumerate(tools):
        print(f"  {i+1}. {tool.get('name')}: {tool.get('description', '')[:100]}...")

    # Simulate user queries and LLM responses
    simulate_queries = [
        "I need wireless headphones with noise cancellation for my commute",
        "Can you recommend kitchen products under $50 with at least 4.5 star ratings?",
        "I'm looking for SoundMaster brand products that are currently in stock",
        "What's the best ergonomic office furniture you have?",
        "I need a gift for someone who enjoys fitness and outdoor activities under $100",
    ]

    for i, query in enumerate(simulate_queries):
        print(f"\n\n{'='*80}")
        print(f"📱 User: {query}")
        print(f"🤖 LLM: I'll search for that information for you...")

        # Call the search tool
        result = client.call_tool("search", {"query": query, "index": INDEX_NAME})

        # Format and present results like an LLM would
        formatted_response = format_search_results(query, result)
        print(f"\n🤖 LLM: {formatted_response}")


def main():
    """Run the enhanced LLM integration demo."""
    print("=== Enhanced Search MCP LLM Integration Demo ===")
    print("This demo shows all steps of the LLM-MCP integration process")

    # Start MCP server
    server_process = start_mcp_server()

    if not server_process:
        print("Failed to start MCP server. Exiting demo.")
        return

    try:
        # Create MCP client
        client = MCPClient(server_process)

        # Simulate enhanced LLM conversation
        simulate_enhanced_llm_conversation(client)

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
