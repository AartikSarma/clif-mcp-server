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
        
        @app.tool()
        async def explore_schema(table_name: str = None) -> str:
            """
            Explore available data tables and their structure.
            
            Args:
                table_name: Optional specific table to explore in detail
                
            Returns:
                Schema information and data statistics
            """
            try:
                result = self.data_explorer.explore_schema(table_name)
                return result
            except Exception as e:
                logger.error(f"Error exploring schema: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def validate_schema() -> str:
            """
            Validate the schema configuration against actual data files.
            
            Returns:
                Validation report with any issues found
            """
            try:
                report = self.data_explorer.validate_schema()
                return json.dumps(report, indent=2)
            except Exception as e:
                logger.error(f"Error validating schema: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def build_cohort(criteria: str) -> str:
            """
            Build a patient cohort based on clinical criteria.
            
            Args:
                criteria: JSON string with cohort criteria including:
                    - age_range: {"min": 18, "max": 65}
                    - icu_los_range: {"min": 3, "max": null}
                    - require_mechanical_ventilation: true/false
                    - medications: ["medication1", "medication2"]
                    - lab_criteria: [{"test_name": "creatinine", "min_value": 2.0}]
                    - custom_filters: [{"table": "labs", "column": "test_name", 
                                      "operator": "contains", "value": "glucose"}]
                    
            Returns:
                Cohort summary with privacy-preserved statistics
            """
            global current_cohort
            
            try:
                criteria_dict = json.loads(criteria)
                cohort, cohort_id = self.cohort_builder.build_cohort(criteria_dict)
                
                # Store for subsequent analyses
                current_cohort = cohort
                
                # Privacy-preserved summary
                summary = {
                    "cohort_id": cohort_id,
                    "size": len(cohort) if len(cohort) >= 10 else "<10 (suppressed)",
                    "criteria": criteria_dict,
                    "available_tables": list(self.schema_loader.discovered_files.keys())
                }
                
                # Save cohort
                cohort_file = self.cohort_builder.save_cohort(cohort, cohort_id, summary)
                summary["saved_to"] = cohort_file
                
                return json.dumps(summary, indent=2)
                
            except Exception as e:
                logger.error(f"Error building cohort: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def analyze_outcomes(outcome_types: str = None) -> str:
            """
            Analyze clinical outcomes for the current cohort.
            
            Args:
                outcome_types: Optional list of specific outcomes to analyze
                
            Returns:
                Privacy-preserved outcome statistics
            """
            global current_cohort
            
            if current_cohort is None:
                return "Error: No cohort loaded. Please build a cohort first."
            
            try:
                if outcome_types:
                    outcomes = json.loads(outcome_types)
                else:
                    outcomes = ["mortality", "icu_los", "hospital_los"]
                
                results = self.outcomes_analyzer.analyze_outcomes(current_cohort, outcomes)
                
                # Apply privacy preservation
                protected_results = self.privacy_guard.protect_output(results)
                
                return json.dumps(protected_results, indent=2)
                
            except Exception as e:
                logger.error(f"Error analyzing outcomes: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def get_schema_info() -> str:
            """
            Get information about the current schema configuration.
            
            Returns:
                Schema name, version, and available tables
            """
            try:
                schema_info = {
                    "name": self.schema_loader.schema.get('name', 'Unknown'),
                    "version": self.schema_loader.schema.get('version', 'Unknown'),
                    "description": self.schema_loader.schema.get('description', ''),
                    "tables": {}
                }
                
                for table_name in self.schema_loader.get_available_tables():
                    table_info = self.schema_loader.get_table_info(table_name)
                    schema_info["tables"][table_name] = {
                        "description": table_info.get('description', ''),
                        "file_pattern": table_info.get('file_pattern', ''),
                        "files_found": len(self.schema_loader.discovered_files.get(table_name, [])),
                        "columns_defined": len(table_info.get('columns', {}))
                    }
                
                return json.dumps(schema_info, indent=2)
                
            except Exception as e:
                logger.error(f"Error getting schema info: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def get_column_values(table_name: str, column_name: str, limit: int = 20) -> str:
            """
            Get unique values and statistics for a specific column.
            
            Args:
                table_name: Name of the table
                column_name: Name of the column
                limit: Maximum number of unique values to return
                
            Returns:
                Column statistics and top values
            """
            try:
                result = self.data_explorer.get_column_values(table_name, column_name, limit)
                return json.dumps(result, indent=2)
            except Exception as e:
                logger.error(f"Error getting column values: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def generate_analysis_code(language: str = "python", target: str = "current_cohort") -> str:
            """
            Generate reproducible analysis code.
            
            Args:
                language: Programming language (python or r)
                target: What to generate code for
                
            Returns:
                Generated code as a string
            """
            global current_cohort
            
            if current_cohort is None and target == "current_cohort":
                return "Error: No cohort loaded. Please build a cohort first."
            
            try:
                # Get schema information for code generation
                schema_info = {
                    "tables": self.schema_loader.get_available_tables(),
                    "schema_name": self.schema_loader.schema.get('name', 'CLIF')
                }
                
                code = self.code_generator.generate_code(
                    analysis_type=target,
                    language=language,
                    cohort_criteria=getattr(current_cohort, 'criteria', {}),
                    schema_info=schema_info
                )
                
                return code
                
            except Exception as e:
                logger.error(f"Error generating code: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def create_derived_variable(variable_config: str) -> str:
            """
            Create a new derived variable based on existing data.
            
            Args:
                variable_config: JSON configuration for the derived variable
                
            Returns:
                Summary of the created variable
            """
            try:
                config = json.loads(variable_config)
                result = self.variable_creator.create_variable(config)
                return json.dumps(result, indent=2)
            except Exception as e:
                logger.error(f"Error creating derived variable: {e}")
                return f"Error: {str(e)}"
        
        @app.tool()
        async def build_ml_model(model_config: str) -> str:
            """
            Build a machine learning model for prediction.
            
            Args:
                model_config: JSON configuration for the model
                
            Returns:
                Model performance metrics and summary
            """
            global current_cohort
            
            if current_cohort is None:
                return "Error: No cohort loaded. Please build a cohort first."
            
            try:
                config = json.loads(model_config)
                result = self.ml_builder.build_model(current_cohort, config)
                
                # Apply privacy preservation
                protected_result = self.privacy_guard.protect_output(result)
                
                return json.dumps(protected_result, indent=2)
                
            except Exception as e:
                logger.error(f"Error building ML model: {e}")
                return f"Error: {str(e)}"


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='CLIF MCP Server with Dynamic Schema Support')
    parser.add_argument('--schema-path', required=True, help='Path to schema configuration JSON file')
    parser.add_argument('--data-path', required=True, help='Path to data directory')
    
    args = parser.parse_args()
    
    try:
        # Initialize server
        server = CLIFServer(args.schema_path, args.data_path)
        
        logger.info(f"Starting CLIF MCP Server")
        logger.info(f"Schema: {args.schema_path}")
        logger.info(f"Data: {args.data_path}")
        logger.info(f"Discovered tables: {list(server.schema_loader.discovered_files.keys())}")
        
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