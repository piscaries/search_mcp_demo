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
from openai import OpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio
from contextlib import AsyncExitStack

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

# Initialize the LLM client
llm_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


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

        # Always print steps 1-3 for clarity
        print(f"\n🔄 STEP 1: LLM decides to use the '{tool_name}' tool")
        print(
            f"🔄 STEP 2: LLM prepares tool call with arguments: {json.dumps(args, indent=2)}"
        )
        print(f"🔄 STEP 3: Sending message to MCP server")

        if self.debug_mode:
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
                        # Always print step 6
                        print(f"🔄 STEP 6: Receiving response from MCP server")

                        if self.debug_mode:
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
                                    # Always print step 4
                                    print("\n🔄 STEP 4: OpenAI generated a query plan")

                                    # Combined approach to extract query plan - handle both formats
                                    plan_found = False

                                    # Try the format with newlines first (most common case)
                                    match = re.search(
                                        r"Query plan:\s*\n([\s\S]*?)(?=\n\nResults:|\Z)",
                                        result,
                                        re.DOTALL,
                                    )
                                    if match:
                                        try:
                                            plan_text = match.group(1).strip()
                                            print(f"📋 Query Plan: {plan_text}")
                                            print(
                                                "\n🔄 STEP 5: Elasticsearch executed the search based on the query plan"
                                            )
                                            plan_found = True
                                        except Exception as e:
                                            print(
                                                f"Error displaying query plan: {str(e)}"
                                            )

                                    # If we still didn't find it, try the inline format
                                    if not plan_found:
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
                                                plan_found = True
                                            except json.JSONDecodeError:
                                                print(
                                                    f"Could not parse query plan: {match.group(1)}"
                                                )

                                    # If we still couldn't find a plan, print step 5 anyway
                                    if not plan_found:
                                        print(
                                            "\n🔄 STEP 5: Elasticsearch executed the search (query plan not displayed)"
                                        )

                                # Always print step 7
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

    # Extract full query plan if available
    query_plan_explanation = ""
    full_query_plan = {}

    # Always extract the query plan from the result
    if "Query plan:" in result:
        # Extract the full query plan JSON
        query_plan_match = re.search(
            r"Query plan:\s*\n(.*?)(?=\n\nResults:|\Z)", result, re.DOTALL
        )

        if query_plan_match:
            try:
                plan_text = query_plan_match.group(1).strip()
                full_query_plan = json.loads(plan_text)
            except Exception as e:
                print(f"Error parsing full query plan: {str(e)}")
                # Try fallback pattern
                try:
                    plan_text = re.sub(r"\s+", " ", plan_text)
                    full_query_plan_str = re.search(r"\{.*\}", plan_text)
                    if full_query_plan_str:
                        full_query_plan = json.loads(full_query_plan_str.group(0))
                except Exception as e2:
                    print(f"Fallback parsing also failed: {str(e2)}")

        # Get the explanation separately (as fallback)
        expl_match = re.search(r'"explanation":\s*"([^"]*)"', result, re.DOTALL)
        if expl_match:
            query_plan_explanation = expl_match.group(1)

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


def real_llm_conversation(client):
    """
    A hybrid approach that combines the reliability of direct MCP client
    with real LLM integration for better stability.
    """
    print("\n=== Real LLM Conversation (Hybrid Mode) ===")
    print("This demonstrates how LLMs can integrate with MCP using a hybrid approach.")

    # First, delete the existing index if it exists
    print("\n[1] INITIALIZING DATA")
    print("🤖 LLM: Let me clean up any existing data...")
    try:
        # Use direct search call
        result = client.call_tool(
            "search", {"query": "DELETE_INDEX", "index": INDEX_NAME}
        )
        print(f"✅ {result}")
    except Exception as e:
        print(f"⚠️ {e}")
        print("Continuing with potentially existing data...")

    # Create the ecommerce test index with fresh data
    print("\n🤖 LLM: Setting up product catalog...")
    result = client.call_tool("create_ecommerce_test_index", {"index": INDEX_NAME})
    print(f"✅ {result}")

    # Get available tools
    print("\n[2] DISCOVERING TOOLS")
    print("🤖 LLM: Discovering available MCP capabilities...")
    tools = client.list_tools()

    if not tools:
        print("❌ No tools available from MCP. Exiting.")
        return

    # Convert MCP tools to OpenAI function format
    openai_functions = []
    for tool in tools:
        # For the search tool, ensure our schema explicitly includes the index parameter
        if tool.get("name") == "search":
            openai_functions.append(
                {
                    "name": "search",
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (supports natural language queries like 'red shoes under $50')",
                            },
                            "index": {
                                "type": "string",
                                "description": "The Elasticsearch index to search",
                                "default": INDEX_NAME,
                            },
                        },
                        "required": ["query"],
                    },
                }
            )
        else:
            openai_functions.append(
                {
                    "name": tool.get("name"),
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                }
            )

    print(f"✅ Discovered {len(tools)} MCP tools")

    # Interactive conversation loop
    print("\n[3] STARTING CONVERSATIONAL LOOP")
    print("\n=== Interactive Mode with Real LLM ===")
    print("Type your product queries below. Type 'exit', 'quit', or 'bye' to end.")

    conversation_history = []  # Keep conversation context

    while True:
        try:
            # Get user input
            user_query = input("\n📱 User: ")
            if user_query.lower() in ["exit", "quit", "bye"]:
                print("🤖 LLM: Goodbye! Thanks for chatting.")
                break

            if not user_query.strip():
                print("🤖 LLM: I didn't catch that. Please try again.")
                continue

            print("🤖 LLM: Processing your request...")

            # Step 1: Update conversation history with user query
            conversation_history.append({"role": "user", "content": user_query})

            # Step 2: Ask LLM to decide which function to call
            print("\n[4] LLM DECIDES ON TOOL USE")
            llm_response = llm_client.chat.completions.create(
                model="gpt-4",
                messages=conversation_history,
                functions=openai_functions,
                function_call="auto",
            )

            # Get LLM's response
            message = llm_response.choices[0].message
            conversation_history.append(message)

            # Print the raw OpenAI response to see what it contains
            print("\n📋 RAW OPENAI RESPONSE (DECISION MAKING)")
            print("-" * 50)
            if hasattr(message, "content") and message.content:
                print(f"Content: {message.content}")
            if hasattr(message, "function_call"):
                print(f"Function Call: {message.function_call}")
                if hasattr(message.function_call, "name"):
                    print(f"Tool Selected: {message.function_call.name}")
                if hasattr(message.function_call, "arguments"):
                    print(f"Parameters: {message.function_call.arguments}")
            print("-" * 50)

            # Check if LLM wants to call a function
            if hasattr(message, "function_call") and message.function_call:
                # Extract function details
                function_name = message.function_call.name
                function_args = json.loads(message.function_call.arguments)

                # Ensure index parameter is set for search queries
                if function_name == "search" and "index" not in function_args:
                    function_args["index"] = INDEX_NAME
                    print("⚠️ Adding missing index parameter to search query")

                print(f"✅ LLM decided to use: {function_name}")
                print(f"   with arguments: {json.dumps(function_args, indent=2)}")

                # Step 3: Call the MCP tool using our reliable direct client
                print("\n[5] EXECUTING MCP TOOL CALL")
                print(f"Calling MCP tool: {function_name}...")

                try:
                    # Use the reliable direct client instead of async session
                    result = client.call_tool(function_name, function_args)

                    # For search specifically, extract and show query plan
                    if function_name == "search" and "Query plan:" in result:
                        print("\n[6] SEARCH QUERY PLANNING")

                        match = re.search(
                            r"Query plan:\s*\n([\s\S]*?)(?=\n\nResults:|\Z)",
                            result,
                            re.DOTALL,
                        )
                        if match:
                            plan_text = match.group(1).strip()
                            print(f"   Query Plan: {plan_text}")

                    print("\n[7] PROCESSING MCP RESPONSE")
                    print("✅ MCP tool execution complete")

                    # Step 4: Add tool response to conversation history
                    conversation_history.append(
                        {"role": "function", "name": function_name, "content": result}
                    )

                    # Step 5: Ask LLM to generate a final response
                    print("\n[8] LLM GENERATES FINAL RESPONSE")
                    final_response = llm_client.chat.completions.create(
                        model="gpt-4",
                        messages=conversation_history,
                    )

                    # Get the final response
                    final_message = final_response.choices[0].message
                    conversation_history.append(final_message)

                    # Display the final response
                    print(f"\n🤖 LLM: {final_message.content}")

                except Exception as e:
                    print(f"❌ Error calling MCP tool: {str(e)}")
                    print(
                        f"🤖 LLM: I encountered an error while trying to search. Let me try another approach."
                    )

                    # Add error to conversation for LLM context
                    conversation_history.append(
                        {
                            "role": "function",
                            "name": function_name,
                            "content": f"Error: {str(e)}",
                        }
                    )

                    # Let LLM generate a response despite the error
                    error_response = llm_client.chat.completions.create(
                        model="gpt-4",
                        messages=conversation_history,
                    )

                    error_message = error_response.choices[0].message
                    conversation_history.append(error_message)
                    print(f"\n🤖 LLM: {error_message.content}")
            else:
                # LLM chose to answer directly
                print("✅ LLM decided to answer directly without using a tool")
                print(f"\n🤖 LLM: {message.content}")

        except Exception as e:
            print(f"❌ Error in conversation loop: {str(e)}")
            print("🤖 LLM: I'm having some technical difficulties. Let's try again.")


def main():
    """Run the enhanced LLM integration demo."""
    print("=== Enhanced Search MCP LLM Integration Demo ===")
    print("This demo shows all steps of the LLM-MCP integration process")

    # Ask the user which mode to run
    print("\nChoose a demo mode:")
    print("1. Simulated LLM conversation (preset queries, no API costs)")
    print("2. Real LLM conversation (interactive, requires OpenAI API key)")

    choice = input("Enter your choice (1 or 2): ")

    if choice == "2":
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            print(
                "Error: OpenAI API key not found. Set it in your .env file or environment variables."
            )
            print("Falling back to simulation mode.")

            # Start MCP server for simulation
            server_process = start_mcp_server()
            if not server_process:
                print("Failed to start MCP server. Exiting demo.")
                return

            try:
                # Create MCP client
                client = MCPClient(server_process)
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
        else:
            # Start MCP server for real LLM hybrid approach
            server_process = start_mcp_server()
            if not server_process:
                print("Failed to start MCP server. Exiting demo.")
                return

            try:
                # Create MCP client
                client = MCPClient(server_process)
                # Use the hybrid approach instead of pure async
                real_llm_conversation(client)
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
    else:
        # Default to simulation mode
        server_process = start_mcp_server()
        if not server_process:
            print("Failed to start MCP server. Exiting demo.")
            return

        try:
            # Create MCP client
            client = MCPClient(server_process)
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
