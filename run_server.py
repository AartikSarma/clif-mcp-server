#!/usr/bin/env python
"""
Entry point for CLIF MCP Server
This wrapper ensures proper module path setup
"""

import sys
import os

# Add the current directory to Python path so modules can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import and run the main server
from server.main import main

if __name__ == "__main__":
    main()