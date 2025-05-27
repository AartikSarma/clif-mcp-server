"""
CLIF MCP Server - Clinical Research Assistant for CLIF Datasets
Focus: Outcomes research with privacy-preserving code generation
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
import os
from pathlib import Path

from mcp import Server, Tool
from mcp.types import TextContent, ToolResult

from .tools.cohort_builder import CohortBuilder
from .tools.outcomes_analyzer import OutcomesAnalyzer
from .tools.code_generator import CodeGenerator
from .tools.data_explorer import DataExplorer
from .security.privacy_guard import PrivacyGuard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIFServer:
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.server = Server("clif-research-server")
        self.privacy_guard = PrivacyGuard()
        
        # Initialize tools
        self.cohort_builder = CohortBuilder(data_path)
        self.outcomes_analyzer = OutcomesAnalyzer(data_path)
        self.code_generator = CodeGenerator()
        self.data_explorer = DataExplorer(data_path)
        
        # Register MCP tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all available MCP tools"""
        
        @self.server.tool()
        async def explore_schema() -> List[Tool]:
            """Explore CLIF dataset schema and summary statistics"""
            return Tool(
                name="explore_schema",
                description="Explore CLIF tables, columns, and data quality metrics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Specific table to explore (optional)"
                        }
                    }
                }
            )
        
        @self.server.tool()
        async def build_cohort() -> List[Tool]:
            """Build patient cohorts for outcomes research"""
            return Tool(
                name="build_cohort",
                description="Define patient cohorts using inclusion/exclusion criteria",
                input_schema={
                    "type": "object",
                    "properties": {
                        "criteria": {
                            "type": "object",
                            "description": "Cohort selection criteria",
                            "properties": {
                                "diagnoses": {"type": "array", "items": {"type": "string"}},
                                "procedures": {"type": "array", "items": {"type": "string"}},
                                "medications": {"type": "array", "items": {"type": "string"}},
                                "age_range": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number"},
                                        "max": {"type": "number"}
                                    }
                                },
                                "icu_los_range": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number"},
                                        "max": {"type": "number"}
                                    }
                                },
                                "require_mechanical_ventilation": {"type": "boolean"},
                                "exclude_readmissions": {"type": "boolean"}
                            }
                        },
                        "save_cohort": {
                            "type": "boolean",
                            "description": "Save cohort definition for reuse"
                        }
                    },
                    "required": ["criteria"]
                }
            )
        
        @self.server.tool()
        async def analyze_outcomes() -> List[Tool]:
            """Analyze clinical outcomes for defined cohorts"""
            return Tool(
                name="analyze_outcomes",
                description="Perform outcomes analysis (mortality, LOS, complications)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cohort_id": {
                            "type": "string",
                            "description": "ID of previously defined cohort"
                        },
                        "outcomes": {
                            "type": "array",
                            "description": "Outcomes to analyze",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "mortality", "icu_mortality", "hospital_mortality",
                                    "icu_los", "hospital_los", "ventilator_days",
                                    "readmission_30d", "aki", "delirium"
                                ]
                            }
                        },
                        "stratify_by": {
                            "type": "array",
                            "description": "Variables to stratify analysis",
                            "items": {"type": "string"}
                        },
                        "adjust_for": {
                            "type": "array",
                            "description": "Confounders to adjust for",
                            "items": {"type": "string"}
                        }
                    },
                    "required": ["outcomes"]
                }
            )
        
        @self.server.tool()
        async def compare_cohorts() -> List[Tool]:
            """Compare outcomes between two cohorts"""
            return Tool(
                name="compare_cohorts",
                description="Statistical comparison of outcomes between cohorts",
                input_schema={
                    "type": "object",
                    "properties": {
                        "cohort1_id": {"type": "string"},
                        "cohort2_id": {"type": "string"},
                        "outcomes": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "adjustment_method": {
                            "type": "string",
                            "enum": ["unadjusted", "regression", "propensity_score", "iptw"]
                        }
                    },
                    "required": ["cohort1_id", "cohort2_id", "outcomes"]
                }
            )
        
        @self.server.tool()
        async def generate_analysis_code() -> List[Tool]:
            """Generate reproducible analysis code"""
            return Tool(
                name="generate_analysis_code",
                description="Generate Python/R code for multi-site validation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "analysis_type": {
                            "type": "string",
                            "description": "Type of analysis performed"
                        },
                        "language": {
                            "type": "string",
                            "enum": ["python", "r"],
                            "default": "python"
                        },
                        "include_visualizations": {
                            "type": "boolean",
                            "default": True
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["script", "notebook"],
                            "default": "script"
                        }
                    },
                    "required": ["analysis_type"]
                }
            )
        
        # Tool implementations
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[ToolResult]:
            logger.info(f"Tool called: {name} with args: {arguments}")
            
            try:
                if name == "explore_schema":
                    result = await self._explore_schema(arguments.get("table_name"))
                elif name == "build_cohort":
                    result = await self._build_cohort(arguments["criteria"], 
                                                    arguments.get("save_cohort", False))
                elif name == "analyze_outcomes":
                    result = await self._analyze_outcomes(
                        arguments.get("cohort_id"),
                        arguments["outcomes"],
                        arguments.get("stratify_by", []),
                        arguments.get("adjust_for", [])
                    )
                elif name == "compare_cohorts":
                    result = await self._compare_cohorts(
                        arguments["cohort1_id"],
                        arguments["cohort2_id"],
                        arguments["outcomes"],
                        arguments.get("adjustment_method", "unadjusted")
                    )
                elif name == "generate_analysis_code":
                    result = await self._generate_code(
                        arguments["analysis_type"],
                        arguments.get("language", "python"),
                        arguments.get("include_visualizations", True),
                        arguments.get("output_format", "script")
                    )
                else:
                    result = f"Unknown tool: {name}"
                
                return [ToolResult(content=[TextContent(text=str(result))])]
                
            except Exception as e:
                logger.error(f"Error in tool {name}: {str(e)}")
                return [ToolResult(
                    content=[TextContent(text=f"Error: {str(e)}")],
                    is_error=True
                )]
    
    async def _explore_schema(self, table_name: Optional[str] = None) -> str:
        """Explore dataset schema with privacy protection"""
        result = self.data_explorer.explore_schema(table_name)
        return self.privacy_guard.sanitize_output(result)
    
    async def _build_cohort(self, criteria: Dict, save_cohort: bool) -> str:
        """Build cohort based on criteria"""
        cohort_df, cohort_id = self.cohort_builder.build_cohort(criteria)
        
        summary = f"Cohort built successfully\n"
        summary += f"Cohort ID: {cohort_id}\n"
        summary += f"Total patients: {len(cohort_df)}\n"
        summary += f"Hospitalizations: {len(cohort_df['hospitalization_id'].unique())}\n"
        
        if save_cohort:
            self.cohort_builder.save_cohort(cohort_id, cohort_df, criteria)
            summary += f"Cohort saved for future use\n"
        
        return summary
    
    async def _analyze_outcomes(self, cohort_id: Optional[str], outcomes: List[str],
                              stratify_by: List[str], adjust_for: List[str]) -> str:
        """Analyze outcomes for a cohort"""
        results = self.outcomes_analyzer.analyze_outcomes(
            cohort_id, outcomes, stratify_by, adjust_for
        )
        
        # Store analysis for code generation
        self.code_generator.store_analysis(results)
        
        return self.outcomes_analyzer.format_results(results)
    
    async def _compare_cohorts(self, cohort1_id: str, cohort2_id: str,
                             outcomes: List[str], adjustment_method: str) -> str:
        """Compare outcomes between two cohorts"""
        results = self.outcomes_analyzer.compare_cohorts(
            cohort1_id, cohort2_id, outcomes, adjustment_method
        )
        
        self.code_generator.store_analysis(results)
        
        return self.outcomes_analyzer.format_comparison(results)
    
    async def _generate_code(self, analysis_type: str, language: str,
                           include_viz: bool, output_format: str) -> str:
        """Generate reproducible analysis code"""
        code = self.code_generator.generate_code(
            analysis_type, language, include_viz, output_format
        )
        
        # Save code to file
        filename = f"clif_analysis_{analysis_type}_{language}.{output_format}"
        output_path = self.data_path.parent / "generated_code" / filename
        output_path.parent.mkdir(exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(code)
        
        return f"Analysis code generated: {output_path}\n\n{code[:500]}..."
    
    async def run(self):
        """Run the MCP server"""
        logger.info("Starting CLIF MCP Server...")
        await self.server.run()


def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="CLIF MCP Server")
    parser.add_argument("--data-path", required=True, help="Path to CLIF CSV files")
    args = parser.parse_args()
    
    # Log startup to stderr for debugging
    print(f"Starting CLIF MCP Server with data path: {args.data_path}", file=sys.stderr)
    
    try:
        server = CLIFServer(args.data_path)
        asyncio.run(server.run())
    except Exception as e:
        print(f"Error starting server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()