#!/usr/bin/env python3
"""
Test script for the Search MCP server.
"""

import subprocess
import time
import threading
import json
import os
import sys
import select


def main():
    print("=== Search MCP Server Test ===")
    print("Starting server process...")

    # Start the server process using Poetry
    process = subprocess.Popen(
        ["poetry", "run", "python", "run_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # Monitor stderr in real-time
    def capture_stderr():
        for line in iter(process.stderr.readline, ""):
            print(f"SERVER ERROR: {line.strip()}")

    stderr_thread = threading.Thread(target=capture_stderr)
    stderr_thread.daemon = True
    stderr_thread.start()

    # Check if process is running
    print(f"Server process started with PID: {process.pid}")
    print(f"Server process is running: {process.poll() is None}")

    # Wait for the ready message
    print("Waiting for ready message...")
    ready_line = process.stdout.readline().strip()
    print(f"Ready message received: {ready_line}")

    try:
        ready_json = json.loads(ready_line)
        if ready_json.get("type") == "ready":
            print("Server is ready to receive messages")
        else:
            print(f"Unexpected ready message type: {ready_json.get('type')}")
    except json.JSONDecodeError:
        print(f"Could not parse ready message as JSON: {ready_line}")

    # Send a list_tools request
    print("\nSending list_tools request to server...")
    message = json.dumps({"id": "test-123", "type": "list_tools"}) + "\n"
    try:
        process.stdin.write(message)
        process.stdin.flush()
        print("Message sent successfully")
    except Exception as e:
        print(f"Error sending message: {e}")

    # Wait for response with timeout
    print("Waiting for server response...")
    start_time = time.time()
    timeout = 10  # seconds

    response_received = False
    while time.time() - start_time < timeout and not response_received:
        # Check if there's data available to read
        readable, _, _ = select.select([process.stdout], [], [], 0.1)
        if readable:
            line = process.stdout.readline().strip()
            if line:
                print(f"Response received: {line}")
                try:
                    response = json.loads(line)
                    print(f"Parsed response: {json.dumps(response, indent=2)}")
                    response_received = True

                    # If we got a list_tools_response, print the tools
                    if response.get("type") == "list_tools_response":
                        tools = response.get("tools", [])
                        print(f"\nFound {len(tools)} tools:")
                        for i, tool in enumerate(tools):
                            print(
                                f"Tool {i+1}: {tool.get('name')} - {tool.get('description', 'No description')[:50]}..."
                            )
                except json.JSONDecodeError:
                    print(f"Could not parse response as JSON: {line}")
        time.sleep(0.1)

    if not response_received:
        print("No response received from server within timeout")

    # Try a tool call
    print("\nSending a tool call request...")
    tool_call = {
        "id": "test-456",
        "type": "tool_call",
        "tool": "search",
        "args": {"query": "headphones", "index": "ecommerce"},
    }
    try:
        process.stdin.write(json.dumps(tool_call) + "\n")
        process.stdin.flush()
        print("Tool call message sent successfully")
    except Exception as e:
        print(f"Error sending tool call message: {e}")

    # Wait for tool call response
    print("Waiting for tool call response...")
    start_time = time.time()

    tool_response_received = False
    while time.time() - start_time < timeout and not tool_response_received:
        readable, _, _ = select.select([process.stdout], [], [], 0.1)
        if readable:
            line = process.stdout.readline().strip()
            if line:
                print(f"Tool call response received: {line}")
                try:
                    response = json.loads(line)
                    print(
                        f"Parsed tool call response: {json.dumps(response, indent=2)[:500]}..."
                    )
                    tool_response_received = True
                except json.JSONDecodeError:
                    print(f"Could not parse tool call response as JSON: {line}")
        time.sleep(0.1)

    if not tool_response_received:
        print("No tool call response received from server within timeout")

    # Clean up
    print("\nTerminating server process...")
    process.terminate()
    try:
        process.wait(timeout=5)
        print("Server process terminated")
    except subprocess.TimeoutExpired:
        print("Server process did not terminate, killing it")
        process.kill()
        process.wait()

    print("\n=== Test Complete ===")


if __name__ == "__main__":
    main()
