#!/usr/bin/env python3
"""
Test MCP server startup and initialization
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from clif_server import CLIFServer

def test_server_initialization():
    """Test that the server initializes correctly"""
    
    print("🚀 Testing MCP Server Initialization")
    print("=" * 40)
    
    schema_path = "schema_config.json"
    data_path = "./data/synthetic"
    
    try:
        # Test server initialization
        print("1️⃣ Initializing server...")
        server = CLIFServer(schema_path, data_path)
        print("   ✅ Server initialized successfully")
        
        # Test schema loader
        print("2️⃣ Testing schema loader...")
        discovered = server.schema_loader.discovered_files
        print(f"   ✅ Found {len(discovered)} tables")
        
        # Test tool initialization  
        print("3️⃣ Testing tool initialization...")
        print(f"   ✅ CohortBuilder: {type(server.cohort_builder).__name__}")
        print(f"   ✅ DataExplorer: {type(server.data_explorer).__name__}")
        print(f"   ✅ SchemaLoader: {type(server.schema_loader).__name__}")
        
        # Test that tools can access data
        print("4️⃣ Testing data access...")
        patient_df = server.schema_loader.load_table('patient')
        if patient_df is not None:
            print(f"   ✅ Can load patient data: {len(patient_df):,} rows")
        else:
            print("   ❌ Cannot load patient data")
            return False
            
        print("\n🎉 Server initialization test passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Server initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_server_initialization()
    if success:
        print("\n✅ The MCP server is ready to use!")
        print("\nTo start the server:")
        print("python3 clif_server.py --schema-path schema_config.json --data-path ./data/synthetic")
    else:
        print("\n❌ Server initialization failed!")
    
    sys.exit(0 if success else 1)