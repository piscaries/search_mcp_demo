#!/usr/bin/env python3
"""
LLM-powered MCP client for the Search MCP server.
"""

import json
import subprocess
import uuid
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()


class MCPClient:
    """Simple MCP client for interacting with the Search MCP server."""

    def __init__(self, server_command: List[str]):
        """
        Initialize the MCP client.

        Args:
            server_command: Command to start the MCP server
        """
        self.process = subprocess.Popen(
            server_command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        # Read the initial message from the server
        self._read_message()

        # Get available tools
        self.tools_info = self.list_tools()
        self.available_tools = {
            tool["name"]: tool for tool in self.tools_info.get("tools", [])
        }

    def _read_message(self) -> Dict[str, Any]:
        """Read a message from the MCP server."""
        line = self.process.stdout.readline().strip()
        if not line:
            return {}
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {line}")
            return {}

    def _send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the MCP server."""
        json_message = json.dumps(message)
        self.process.stdin.write(json_message + "\n")
        self.process.stdin.flush()

    def list_tools(self) -> Dict[str, Any]:
        """List the available tools from the MCP server."""
        message = {"id": str(uuid.uuid4()), "type": "list_tools"}
        self._send_message(message)
        return self._read_message()

    def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            **kwargs: Arguments to pass to the tool

        Returns:
            The response from the MCP server
        """
        message = {
            "id": str(uuid.uuid4()),
            "type": "tool_call",
            "tool": tool_name,
            "args": kwargs,
        }
        self._send_message(message)
        return self._read_message()

    def close(self) -> None:
        """Close the connection to the MCP server."""
        if self.process:
            self.process.terminate()
            self.process.wait()


class LLMPoweredMCPClient:
    """LLM-powered MCP client that uses an LLM to decide which tools to call."""

    def __init__(self, server_command: List[str]):
        """
        Initialize the LLM-powered MCP client.

        Args:
            server_command: Command to start the MCP server
        """
        self.mcp_client = MCPClient(server_command)
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # Create a system prompt with tool descriptions
        self.system_prompt = self._create_system_prompt()

        # Conversation history
        self.conversation_history = [{"role": "system", "content": self.system_prompt}]

    def _create_system_prompt(self) -> str:
        """Create a system prompt with tool descriptions."""
        tools_info = self.mcp_client.tools_info.get("tools", [])

        tool_descriptions = "\n\n".join(
            [
                f"Tool: {tool['name']}\n"
                f"Description: {tool.get('description', 'No description')}\n"
                f"Parameters: {json.dumps(tool.get('parameters', {}), indent=2)}"
                for tool in tools_info
            ]
        )

        return f"""You are an AI assistant that helps users search for products in an e-commerce system.
You have access to the following tools through an MCP server:

{tool_descriptions}

When a user asks a question, analyze it and decide which tool to call.
Format your response as a JSON object with the following structure:
{{
  "thought": "Your reasoning about what tool to use and why",
  "tool": "The name of the tool to call",
  "parameters": {{
    "param1": "value1",
    "param2": "value2",
    ...
  }}
}}

If you need to have a conversation with the user without calling a tool, respond with:
{{
  "thought": "Your reasoning",
  "message": "Your message to the user"
}}

Always think carefully about which tool is most appropriate for the user's request.
"""

    def process_query(self, user_query: str) -> str:
        """
        Process a user query using the LLM to decide which tool to call.

        Args:
            user_query: The user's query

        Returns:
            The response to the user
        """
        # Add user query to conversation history
        self.conversation_history.append({"role": "user", "content": user_query})

        # Get LLM response
        llm_response = self.openai_client.chat.completions.create(
            model=self.model, messages=self.conversation_history, temperature=0.1
        )

        llm_content = llm_response.choices[0].message.content

        # Add LLM response to conversation history
        self.conversation_history.append({"role": "assistant", "content": llm_content})

        # Parse LLM response
        try:
            response_json = json.loads(llm_content)

            # If the LLM wants to send a message without calling a tool
            if "message" in response_json:
                return response_json["message"]

            # If the LLM wants to call a tool
            if "tool" in response_json and "parameters" in response_json:
                tool_name = response_json["tool"]
                parameters = response_json["parameters"]

                # Call the tool
                tool_response = self.mcp_client.call_tool(tool_name, **parameters)

                # Add tool response to conversation history
                self.conversation_history.append(
                    {
                        "role": "system",
                        "content": f"Tool response: {tool_response.get('result', 'No result')}",
                    }
                )

                return tool_response.get("result", "Error: No result from tool")

            return "Error: Invalid LLM response format"

        except json.JSONDecodeError:
            return "Error: Could not parse LLM response as JSON"

    def close(self) -> None:
        """Close the connection to the MCP server."""
        self.mcp_client.close()


def run_llm_powered_client():
    """Run the LLM-powered MCP client."""
    print("LLM-Powered E-commerce Search Assistant")
    print("=======================================\n")

    # Start the LLM-powered MCP client
    client = LLMPoweredMCPClient(["python", "-m", "search_mcp_pkg.search_mcp"])

    try:
        # Check if the index exists and create it if it doesn't
        index_check_response = client.process_query(
            "Do we have any product data? If not, please create some test data."
        )
        print(f"Initialization: {index_check_response}\n")

        # Main interaction loop
        print("You can now ask questions about products. Type 'exit' to quit.\n")

        while True:
            user_query = input("You: ")

            if user_query.lower() in ["exit", "quit", "bye"]:
                print("Exiting...")
                break

            response = client.process_query(user_query)
            print(f"\nAssistant: {response}\n")

    finally:
        # Close the client
        client.close()


if __name__ == "__main__":
    run_llm_powered_client()
