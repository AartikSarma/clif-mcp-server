#!/usr/bin/env python
"""Test data dictionary functionality"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from clif_server import CLIFServer

# Initialize server
server = CLIFServer("./data/synthetic")

# Get variable info
all_info = server.comprehensive_dictionary.get_variable_info()

print("Variable Info Structure:")
print("-" * 50)

if all_info:
    for table_name, variables in all_info.items():
        print(f"\n{table_name}:")
        if isinstance(variables, dict):
            print(f"  Number of variables: {len(variables)}")
            if variables:
                # Show first few variables
                for i, (var_name, var_info) in enumerate(list(variables.items())[:3]):
                    print(f"  - {var_name}: {var_info}")
                    if i == 2 and len(variables) > 3:
                        print(f"  ... and {len(variables) - 3} more variables")
        else:
            print(f"  Value type: {type(variables)}")
else:
    print("No variable info returned!")

# Test the actual tool output
print("\n\nSimulating show_data_dictionary tool:")
print("-" * 50)

# Build data dictionary first
server._build_data_dictionary()

# Get fresh data
all_info = server.comprehensive_dictionary.get_variable_info()

# Format like the tool does
result = "COMPREHENSIVE CLIF DATA DICTIONARY\n"
result += "=" * 50 + "\n"

# Show regular tables
table_count = 0
for table_name, variables in all_info.items():
    if isinstance(variables, dict) and table_name != 'derived_variables':
        table_count += 1
        result += f"\n{table_name.upper()} TABLE:\n"
        result += "-" * 30 + "\n"
        var_count = 0
        if variables:
            for var_name, var_info in variables.items():
                if isinstance(var_info, dict):
                    var_type = var_info.get('type', 'unknown')
                    desc = var_info.get('description', '')
                    result += f"  • {var_name} ({var_type}): {desc}\n"
                    var_count += 1
                    if var_count >= 3:  # Limit output for testing
                        result += f"  ... and {len(variables) - 3} more variables\n"
                        break
        else:
            result += "  (No variables found in this table)\n"

print(f"Found {table_count} tables")
print(result[:1000] + "..." if len(result) > 1000 else result)