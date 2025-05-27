#!/usr/bin/env python
"""
Quick start script for CLIF MCP Server
Generates synthetic data and provides usage examples
"""

import os
import sys
from pathlib import Path

def main():
    print("CLIF MCP Server Quick Start")
    print("=" * 50)
    
    # Check if synthetic data exists
    data_dir = Path("data/synthetic")
    
    if not data_dir.exists() or not any(data_dir.glob("*.csv")):
        print("\n1. Generating synthetic CLIF dataset...")
        os.system(f"{sys.executable} synthetic_data/clif_generator.py")
    else:
        print("\n1. Synthetic data already exists")
    
    print("\n2. To start the MCP server, run:")
    print(f"   {sys.executable} -m server.main --data-path ./data/synthetic")
    
    print("\n3. Add to Claude Desktop config (~/.claude/claude_desktop_config.json):")
    print('''
{
  "mcpServers": {
    "clif": {
      "command": "python",
      "args": ["-m", "server.main", "--data-path", "%s"],
      "cwd": "%s"
    }
  }
}''' % (str(Path.cwd() / "data" / "synthetic"), str(Path.cwd())))
    
    print("\n4. Example queries to try in Claude Desktop:")
    print("   - 'Explore the CLIF schema'")
    print("   - 'Build a cohort of ventilated patients over 65'")
    print("   - 'Analyze mortality for the ICU population'")
    print("   - 'Compare outcomes between short and long ICU stays'")
    print("   - 'Generate Python code for mortality analysis'")
    
    print("\n5. For development:")
    print(f"   - Synthetic data location: {data_dir.absolute()}")
    print("   - Modify cohort criteria in: server/tools/cohort_builder.py")
    print("   - Add outcomes in: server/tools/outcomes_analyzer.py")
    print("   - Adjust privacy settings in: server/security/privacy_guard.py")

if __name__ == "__main__":
    main()