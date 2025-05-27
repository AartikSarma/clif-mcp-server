#!/usr/bin/env python
"""
Simple test MCP server to verify setup
"""

import sys
import json

def main():
    # Log to stderr for debugging
    print("Test MCP Server starting...", file=sys.stderr)
    
    # Simple JSON-RPC loop
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
                
            # Parse request
            request = json.loads(line)
            print(f"Received request: {request}", file=sys.stderr)
            
            # Handle initialize
            if request.get("method") == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {
                            "name": "clif-test-server",
                            "version": "0.1.0"
                        }
                    }
                }
                print(json.dumps(response))
                sys.stdout.flush()
                
            # Handle other methods
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "error": {
                        "code": -32601,
                        "message": "Method not found"
                    }
                }
                print(json.dumps(response))
                sys.stdout.flush()
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            break

if __name__ == "__main__":
    main()