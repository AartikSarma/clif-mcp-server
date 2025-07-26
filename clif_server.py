#!/usr/bin/env python
"""
CLIF MCP Server - Dynamic Schema Based Implementation
Supports flexible database schemas through configuration files
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
import argparse

import pandas as pd

from mcp.server import Server  # type: ignore
from mcp.server.stdio import stdio_server  # type: ignore
from mcp.types import Tool, TextContent  # type: ignore

# Import tool classes
from server.tools.cohort_builder import CohortBuilder
from server.tools.data_explorer import DataExplorer
from server.tools.schema_loader import SchemaLoader
from server.tools.outcomes_analyzer import OutcomesAnalyzer
from server.tools.code_generator import CodeGenerator
from server.tools.ml_model_builder import MLModelBuilder
from server.tools.derived_variable_creator import DerivedVariableCreator
from server.security.privacy_guard import PrivacyGuard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app = Server("clif-mcp-server")
current_cohort = None
last_regression_config = None


class CLIFServer:
    """Main CLIF MCP Server with dynamic schema support"""
    
    def __init__(self, schema_path: str, data_path: str):
        self.schema_path = Path(schema_path)
        self.data_path = Path(data_path)
        
        # Validate paths
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")
        
        # Initialize schema loader
        self.schema_loader = SchemaLoader(schema_path, data_path)
        
        # Initialize tools with dynamic schema
        self.cohort_builder = CohortBuilder(schema_path, data_path)
        self.data_explorer = DataExplorer(schema_path, data_path)
        
        # These tools will be updated to use dynamic schema
        self.outcomes_analyzer = OutcomesAnalyzer(data_path)
        self.code_generator = CodeGenerator()
        self.ml_builder = MLModelBuilder(data_path)
        self.variable_creator = DerivedVariableCreator(data_path)
        
        # Initialize privacy guard
        self.privacy_guard = PrivacyGuard()
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all available tools"""
        pass  # Tools are registered at module level


@app.list_tools()
async def list_available_tools() -> list[Tool]:
    """List all available tools for the CLIF MCP server"""
    return [
        Tool(
            name="explore_schema",
            description="Explore available data tables and their structure",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Optional specific table to explore in detail"
                    }
                }
            }
        ),
        Tool(
            name="validate_schema",
            description="Validate the schema configuration against actual data files",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="build_cohort",
            description="Build a patient cohort based on clinical criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "string",
                        "description": "JSON string with cohort criteria"
                    }
                },
                "required": ["criteria"]
            }
        ),
        Tool(
            name="analyze_outcomes",
            description="Analyze clinical outcomes for the current cohort",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome_types": {
                        "type": "string",
                        "description": "Optional list of specific outcomes to analyze"
                    }
                }
            }
        ),
        Tool(
            name="get_schema_info",
            description="Get information about the current schema configuration",
            inputSchema={"type": "object", "properties": {}}
        ),
        Tool(
            name="get_column_values",
            description="Get unique values and statistics for a specific column",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {"type": "string", "description": "Name of the table"},
                    "column_name": {"type": "string", "description": "Name of the column"},
                    "limit": {"type": "integer", "description": "Maximum number of unique values to return"}
                },
                "required": ["table_name", "column_name"]
            }
        ),
        Tool(
            name="generate_analysis_code",
            description="Generate reproducible analysis code",
            inputSchema={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "description": "Programming language (python or r)"},
                    "target": {"type": "string", "description": "What to generate code for"}
                }
            }
        )
    ]


# Global server instance (will be set in main)
server_instance = None


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    global server_instance, current_cohort
    
    if server_instance is None:
        return [TextContent(type="text", text="Error: Server not initialized")]
    
    try:
        if name == "explore_schema":
            table_name = arguments.get("table_name")
            result = server_instance.data_explorer.explore_schema(table_name)
            return [TextContent(type="text", text=result)]
            
        elif name == "validate_schema":
            report = server_instance.data_explorer.validate_schema()
            return [TextContent(type="text", text=json.dumps(report, indent=2))]
            
        elif name == "build_cohort":
            criteria_str = arguments.get("criteria", "{}")
            criteria_dict = json.loads(criteria_str)
            cohort, cohort_id = server_instance.cohort_builder.build_cohort(criteria_dict)
            
            # Store for subsequent analyses
            current_cohort = cohort
            
            # Privacy-preserved summary
            summary = {
                "cohort_id": cohort_id,
                "size": len(cohort) if len(cohort) >= 10 else "<10 (suppressed)",
                "criteria": criteria_dict,
                "available_tables": list(server_instance.schema_loader.discovered_files.keys())
            }
            
            # Save cohort
            cohort_file = server_instance.cohort_builder.save_cohort(cohort, cohort_id, summary)
            summary["saved_to"] = cohort_file
            
            return [TextContent(type="text", text=json.dumps(summary, indent=2))]
            
        elif name == "analyze_outcomes":
            if current_cohort is None:
                return [TextContent(type="text", text="Error: No cohort loaded. Please build a cohort first.")]
            
            outcome_types_str = arguments.get("outcome_types")
            if outcome_types_str:
                outcomes = json.loads(outcome_types_str)
            else:
                outcomes = ["mortality", "icu_los", "hospital_los"]
            
            results = server_instance.outcomes_analyzer.analyze_outcomes(current_cohort, outcomes)
            
            # Apply privacy preservation
            protected_results = server_instance.privacy_guard.protect_output(results)
            
            return [TextContent(type="text", text=json.dumps(protected_results, indent=2))]
            
        elif name == "get_schema_info":
            schema_info = {
                "name": server_instance.schema_loader.schema.get('name', 'Unknown'),
                "version": server_instance.schema_loader.schema.get('version', 'Unknown'),
                "description": server_instance.schema_loader.schema.get('description', ''),
                "tables": {}
            }
            
            for table_name in server_instance.schema_loader.get_available_tables():
                table_info = server_instance.schema_loader.get_table_info(table_name)
                schema_info["tables"][table_name] = {
                    "description": table_info.get('description', ''),
                    "file_pattern": table_info.get('file_pattern', ''),
                    "files_found": len(server_instance.schema_loader.discovered_files.get(table_name, [])),
                    "columns_defined": len(table_info.get('columns', {}))
                }
            
            return [TextContent(type="text", text=json.dumps(schema_info, indent=2))]
            
        elif name == "get_column_values":
            table_name = arguments.get("table_name")
            column_name = arguments.get("column_name")
            limit = arguments.get("limit", 20)
            
            result = server_instance.data_explorer.get_column_values(table_name, column_name, limit)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
        elif name == "generate_analysis_code":
            language = arguments.get("language", "python")
            target = arguments.get("target", "current_cohort")
            
            if current_cohort is None and target == "current_cohort":
                return [TextContent(type="text", text="Error: No cohort loaded. Please build a cohort first.")]
            
            # Get schema information for code generation
            schema_info = {
                "tables": server_instance.schema_loader.get_available_tables(),
                "schema_name": server_instance.schema_loader.schema.get('name', 'CLIF')
            }
            
            code = server_instance.code_generator.generate_code(
                analysis_type=target,
                language=language,
                cohort_criteria=getattr(current_cohort, 'criteria', {}),
                schema_info=schema_info
            )
            
            return [TextContent(type="text", text=code)]
            
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        import traceback
        error_msg = f"Error in {name}: {str(e)}\n{traceback.format_exc()}"
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Main entry point"""
    global server_instance
    
    parser = argparse.ArgumentParser(description='CLIF MCP Server with Dynamic Schema Support')
    parser.add_argument('--schema-path', required=True, help='Path to schema configuration JSON file')
    parser.add_argument('--data-path', required=True, help='Path to data directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize server
        server_instance = CLIFServer(args.schema_path, args.data_path)
        
        logger.info(f"Starting CLIF MCP Server")
        logger.info(f"Schema: {args.schema_path}")
        logger.info(f"Data: {args.data_path}")
        logger.info(f"Discovered tables: {list(server_instance.schema_loader.discovered_files.keys())}")
        
        # Run server
        async with stdio_server() as streams:
            await app.run(
                streams[0],
                streams[1],
                app.create_initialization_options()
            )
            
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())