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

from mcp.server import Server
from mcp.types import Tool, TextContent, ToolResult

from .tools.cohort_builder import CohortBuilder
from .tools.outcomes_analyzer import OutcomesAnalyzer
from .tools.code_generator import CodeGenerator
from .tools.data_explorer import DataExplorer
from .tools.ml_model_builder import MLModelBuilder
from .tools.comprehensive_dictionary import ComprehensiveDictionary
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
        self.ml_model_builder = MLModelBuilder(data_path)
        self.comprehensive_dictionary = ComprehensiveDictionary(data_path)
        
        # Register MCP tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all available MCP tools"""
        
        @self.server.tool(
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
        async def explore_schema(table_name: Optional[str] = None) -> str:
            """Explore CLIF dataset schema and summary statistics"""
            result = self.data_explorer.explore_schema(table_name)
            return self.privacy_guard.sanitize_output(result)
        
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
        
        @self.server.tool()
        async def get_variable_dictionary() -> List[Tool]:
            """Get information about available variables in the dataset"""
            return Tool(
                name="get_variable_dictionary",
                description="Get dynamic information about variables available in the CLIF dataset",
                input_schema={
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Specific table to get variables for (optional)"
                        },
                        "variable_name": {
                            "type": "string",
                            "description": "Specific variable to get information about (optional)"
                        },
                        "analysis_type": {
                            "type": "string",
                            "description": "Get suggested variables for analysis type (mortality_prediction, los_prediction, delirium_risk)",
                            "enum": ["mortality_prediction", "los_prediction", "delirium_risk"]
                        }
                    }
                }
            )
        
        @self.server.tool()
        async def build_ml_model() -> List[Tool]:
            """Build machine learning models for clinical prediction"""
            return Tool(
                name="build_ml_model",
                description="Build flexible ML models for predicting clinical outcomes",
                input_schema={
                    "type": "object",
                    "properties": {
                        "outcome": {
                            "type": "string",
                            "description": "Target variable to predict (e.g., mortality, los, readmission)"
                        },
                        "features": {
                            "type": "array",
                            "description": "List of feature columns to use",
                            "items": {"type": "string"}
                        },
                        "model_type": {
                            "type": "string",
                            "description": "Model type (auto, logistic_regression, random_forest, etc.)",
                            "default": "auto"
                        },
                        "cohort_id": {
                            "type": "string",
                            "description": "Optional cohort ID to use for modeling"
                        },
                        "feature_selection": {
                            "type": "object",
                            "description": "Feature selection configuration",
                            "properties": {
                                "method": {
                                    "type": "string",
                                    "enum": ["kbest", "mutual_info", "rfe"]
                                },
                                "n_features": {"type": "integer"}
                            }
                        },
                        "hyperparameter_tuning": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to tune hyperparameters"
                        },
                        "validation_split": {
                            "type": "number",
                            "default": 0.2,
                            "description": "Test set proportion"
                        }
                    },
                    "required": ["outcome", "features"]
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
                elif name == "get_variable_dictionary":
                    result = await self._get_variable_dictionary(
                        arguments.get("table_name"),
                        arguments.get("variable_name"),
                        arguments.get("analysis_type")
                    )
                elif name == "build_ml_model":
                    result = await self._build_ml_model(
                        arguments["outcome"],
                        arguments["features"],
                        arguments.get("model_type", "auto"),
                        arguments.get("cohort_id"),
                        arguments.get("feature_selection"),
                        arguments.get("hyperparameter_tuning", True),
                        arguments.get("validation_split", 0.2)
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
    
    async def _build_ml_model(self, outcome: str, features: List[str],
                             model_type: str, cohort_id: Optional[str],
                             feature_selection: Optional[Dict],
                             hyperparameter_tuning: bool,
                             validation_split: float) -> str:
        """Build ML model for clinical prediction"""
        
        # Load data
        if cohort_id:
            # Load specific cohort
            df = self.cohort_builder.load_cohort(cohort_id)
        else:
            # Load all patient data
            patient_df = pd.read_csv(self.data_path / "patient.csv")
            hospitalization_df = pd.read_csv(self.data_path / "hospitalization.csv")
            
            # Merge for basic features
            df = patient_df.merge(hospitalization_df, on="patient_id")
        
        # Build model
        result = self.ml_model_builder.build_model(
            df=df,
            outcome=outcome,
            features=features,
            model_type=model_type,
            feature_selection=feature_selection,
            hyperparameter_tuning=hyperparameter_tuning,
            validation_split=validation_split
        )
        
        # Format results with privacy protection
        summary = f"ML Model Built Successfully\n"
        summary += f"Model ID: {result['model_id']}\n"
        summary += f"Model Type: {result['model_type']}\n"
        summary += f"Task: {result['task']}\n"
        summary += f"Number of samples: {result['n_samples']}\n"
        summary += f"Number of features: {result['n_features']}\n\n"
        
        summary += "Performance Metrics:\n"
        summary += "Training:\n"
        for metric, value in result['performance']['train'].items():
            summary += f"  - {metric}: {value:.3f}\n"
        
        summary += "\nTest:\n"
        for metric, value in result['performance']['test'].items():
            summary += f"  - {metric}: {value:.3f}\n"
        
        # Feature importance (top 10)
        if result['feature_importance']:
            summary += "\nTop 10 Important Features:\n"
            for i, (feature, importance) in enumerate(list(result['feature_importance'].items())[:10]):
                summary += f"  {i+1}. {feature}: {importance:.3f}\n"
        
        # Generate model report
        model_report = self.ml_model_builder.generate_model_report(result['model_id'])
        
        # Store for code generation
        self.code_generator.store_analysis({
            'type': 'ml_model',
            'model_info': result,
            'report': model_report
        })
        
        return self.privacy_guard.sanitize_output(summary)
    
    async def _get_variable_dictionary(self, table_name: Optional[str] = None,
                                      variable_name: Optional[str] = None,
                                      analysis_type: Optional[str] = None) -> str:
        """Get dynamic variable information from the dataset"""
        
        if analysis_type:
            # Get suggested variables for analysis type
            suggestions = self.data_explorer.suggest_analysis_variables(analysis_type)
            output = f"Suggested variables for {analysis_type}:\n\n"
            
            if suggestions.get("outcome"):
                output += f"Outcome variable: {suggestions['outcome']}\n"
            else:
                output += f"Outcome variable: Not available\n"
                
            if suggestions.get("predictors"):
                output += f"Predictor variables:\n"
                for predictor in suggestions["predictors"]:
                    output += f"  - {predictor}\n"
            else:
                output += "No suitable predictor variables found\n"
                
            if suggestions.get("note"):
                output += f"\nNote: {suggestions['note']}\n"
                
            return output
            
        elif variable_name and table_name:
            # Get specific variable info
            info = self.data_explorer.get_variable_dictionary(table_name, variable_name)
            if not info:
                return f"Variable '{variable_name}' not found in table '{table_name}'"
                
            output = f"Variable: {table_name}.{variable_name}\n"
            output += f"Description: {info.get('description', 'N/A')}\n"
            output += f"Type: {info.get('type', 'unknown')}\n"
            
            if info.get('values'):
                output += f"Values: {', '.join(str(v) for v in info['values'][:10])}"
                if len(info['values']) > 10:
                    output += f" ... and {len(info['values']) - 10} more"
                output += "\n"
                
            if info.get('min') is not None:
                output += f"Range: {info['min']} - {info['max']}\n"
                
            if info.get('completeness') is not None:
                output += f"Completeness: {info['completeness']*100:.1f}%\n"
                
            return output
            
        elif table_name:
            # Get all variables for a table
            info = self.data_explorer.get_variable_dictionary(table_name)
            if not info:
                return f"Table '{table_name}' not found"
                
            output = f"Variables in {table_name} table:\n\n"
            for var_name, var_info in info.items():
                output += f"  {var_name}:\n"
                output += f"    Type: {var_info.get('type', 'unknown')}\n"
                output += f"    Description: {var_info.get('description', 'N/A')}\n"
                if var_info.get('completeness') is not None:
                    output += f"    Completeness: {var_info['completeness']*100:.1f}%\n"
                output += "\n"
                
            return output
            
        else:
            # List all available variables
            variables = self.data_explorer.list_available_variables()
            output = "Available variables in CLIF dataset:\n\n"
            
            for table, var_list in variables.items():
                if var_list:
                    output += f"{table.upper()}:\n"
                    for var in sorted(var_list):
                        output += f"  - {var}\n"
                    output += "\n"
                    
            return output
    
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