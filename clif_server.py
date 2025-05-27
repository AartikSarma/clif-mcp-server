#!/usr/bin/env python
"""
CLIF MCP Server - Consolidated Implementation
Combines the best features from both implementations with clean architecture
"""

import asyncio
import logging
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import argparse

import pandas as pd
import numpy as np

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import all tool classes
from server.tools.cohort_builder import CohortBuilder
from server.tools.outcomes_analyzer import OutcomesAnalyzer
from server.tools.code_generator import CodeGenerator
from server.tools.data_explorer import DataExplorer
from server.tools.ml_model_builder import MLModelBuilder
from server.tools.comprehensive_dictionary import DynamicCLIFDictionary as ComprehensiveDictionary
from server.security.privacy_guard import PrivacyGuard

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app = Server("clif-mcp-server")
current_cohort = None
last_regression_config = None
data_dictionary = {}


class CLIFServer:
    """Main CLIF MCP Server with consolidated functionality"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        
        # Initialize tools
        self.cohort_builder = CohortBuilder(data_path)
        self.outcomes_analyzer = OutcomesAnalyzer(data_path)
        self.code_generator = CodeGenerator()
        self.data_explorer = DataExplorer(data_path)
        self.ml_model_builder = MLModelBuilder(data_path)
        self.comprehensive_dictionary = ComprehensiveDictionary(data_path)
        self.privacy_guard = PrivacyGuard()
        
        # Build data dictionary on startup
        self._build_data_dictionary()
        
    def _build_data_dictionary(self):
        """Build comprehensive data dictionary from actual data"""
        global data_dictionary
        
        try:
            # Get the dictionary info from comprehensive_dictionary
            data_dictionary = self.comprehensive_dictionary.get_variable_info()
            
            # Add actual categorical values from data
            categorical_values = {}
            for table_name in self.data_explorer.tables:
                try:
                    df = pd.read_csv(self.data_path / f"{table_name}.csv", nrows=1000)
                    for col in df.select_dtypes(include=['object']).columns:
                        unique_vals = df[col].dropna().unique()
                        categorical_values[f"{table_name}.{col}"] = list(unique_vals)[:20]
                except Exception as e:
                    logger.warning(f"Could not read {table_name}: {e}")
            
            data_dictionary["actual_categorical_values"] = categorical_values
            data_dictionary["available_tables"] = self.data_explorer.tables
            
            logger.info(f"Data dictionary built with {len(data_dictionary)} entries")
            
        except Exception as e:
            logger.error(f"Error building data dictionary: {e}")
            data_dictionary = {"error": "Could not build data dictionary"}


# Initialize server instance
clif_server = None


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available CLIF analysis tools"""
    
    return [
        Tool(
            name="show_data_dictionary",
            description="Show comprehensive data dictionary with all available variables",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="explore_data",
            description="Explore CLIF dataset schema and summary statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "Specific table to explore (optional)"
                    }
                }
            }
        ),
        Tool(
            name="build_cohort",
            description="Build patient cohort with flexible criteria",
            inputSchema={
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "object",
                        "description": "Cohort selection criteria",
                        "properties": {
                            "age_range": {
                                "type": "object",
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"}
                                }
                            },
                            "require_mechanical_ventilation": {"type": "boolean"},
                            "icu_los_range": {
                                "type": "object",
                                "properties": {
                                    "min": {"type": "number"},
                                    "max": {"type": "number"}
                                }
                            },
                            "exclude_readmissions": {"type": "boolean"},
                            "diagnoses": {"type": "array", "items": {"type": "string"}},
                            "medications": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "save_cohort": {
                        "type": "boolean",
                        "description": "Save cohort for reuse"
                    }
                },
                "required": ["criteria"]
            }
        ),
        Tool(
            name="analyze_outcomes",
            description="Analyze clinical outcomes for cohorts",
            inputSchema={
                "type": "object",
                "properties": {
                    "cohort_id": {
                        "type": "string",
                        "description": "ID of previously saved cohort (optional)"
                    },
                    "outcomes": {
                        "type": "array",
                        "description": "Outcomes to analyze (e.g., mortality, icu_los, hospital_los, ventilator_days, or any numeric column)",
                        "items": {"type": "string"}
                    },
                    "stratify_by": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["outcomes"]
            }
        ),
        Tool(
            name="compare_cohorts",
            description="Compare outcomes between two cohorts",
            inputSchema={
                "type": "object",
                "properties": {
                    "cohort1_id": {"type": "string"},
                    "cohort2_id": {"type": "string"},
                    "outcomes": {
                        "type": "array",
                        "description": "Outcomes to compare between cohorts",
                        "items": {"type": "string"}
                    },
                    "adjustment_method": {
                        "type": "string",
                        "enum": ["unadjusted", "regression", "propensity_score", "iptw"]
                    }
                },
                "required": ["cohort1_id", "cohort2_id", "outcomes"]
            }
        ),
        Tool(
            name="fit_regression",
            description="Fit various regression models (linear, logistic, poisson, negative_binomial, cox, mixed)",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome": {"type": "string"},
                    "predictors": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "model_type": {
                        "type": "string",
                        "enum": ["linear", "logistic", "poisson", "negative_binomial", "cox", "mixed_linear"]
                    },
                    "interaction_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Interaction terms like 'age*sex'"
                    },
                    "random_effects": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Random effects for mixed models"
                    }
                },
                "required": ["outcome", "predictors", "model_type"]
            }
        ),
        Tool(
            name="descriptive_stats",
            description="Generate descriptive statistics for variables",
            inputSchema={
                "type": "object",
                "properties": {
                    "variables": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "stratify_by": {
                        "type": "string",
                        "description": "Variable to stratify by"
                    }
                },
                "required": ["variables"]
            }
        ),
        Tool(
            name="build_ml_model",
            description="Build machine learning models for prediction",
            inputSchema={
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Target variable to predict"
                    },
                    "features": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Feature columns for prediction"
                    },
                    "model_type": {
                        "type": "string",
                        "description": "Model type (auto, logistic_regression, random_forest, xgboost, etc.)",
                        "default": "auto"
                    },
                    "feature_selection": {
                        "type": "object",
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
                        "default": True
                    }
                },
                "required": ["outcome", "features"]
            }
        ),
        Tool(
            name="generate_code",
            description="Generate reproducible analysis code",
            inputSchema={
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "r"]
                    },
                    "include_all": {
                        "type": "boolean",
                        "description": "Include all analyses performed"
                    }
                },
                "required": ["language"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute CLIF analysis tools"""
    global current_cohort, last_regression_config
    
    try:
        if name == "show_data_dictionary":
            # Ensure data dictionary is built
            if not data_dictionary:
                clif_server._build_data_dictionary()
            
            # Get fresh data from comprehensive dictionary
            all_info = clif_server.comprehensive_dictionary.get_variable_info()
            
            result = "COMPREHENSIVE CLIF DATA DICTIONARY\n"
            result += "=" * 50 + "\n"
            
            # Show regular tables
            for table_name, variables in all_info.items():
                if isinstance(variables, dict) and table_name != 'derived_variables':
                    result += f"\n{table_name.upper()} TABLE:\n"
                    result += "-" * 30 + "\n"
                    if variables:
                        for var_name, var_info in variables.items():
                            if isinstance(var_info, dict):
                                var_type = var_info.get('type', 'unknown')
                                desc = var_info.get('description', '')
                                result += f"  • {var_name} ({var_type}): {desc}\n"
                    else:
                        result += "  (No variables found in this table)\n"
            
            # Show derived variables
            if 'derived_variables' in all_info:
                result += "\nDERIVED VARIABLES:\n"
                result += "-" * 30 + "\n"
                for var_name, var_info in all_info['derived_variables'].items():
                    if isinstance(var_info, dict) and var_info.get('available', False):
                        var_type = var_info.get('type', 'unknown')
                        desc = var_info.get('description', '')
                        result += f"  • {var_name} ({var_type}): {desc}\n"
            
            # Add categorical values if available
            if data_dictionary and 'actual_categorical_values' in data_dictionary:
                result += "\nCATEGORICAL VALUE EXAMPLES:\n"
                result += "-" * 30 + "\n"
                for var_path, values in list(data_dictionary['actual_categorical_values'].items())[:10]:
                    if values:
                        result += f"  • {var_path}: {', '.join(str(v) for v in values[:5])}\n"
                        if len(values) > 5:
                            result += f"    ... and {len(values) - 5} more values\n"
            
            if not all_info or (isinstance(all_info, dict) and not any(v for k, v in all_info.items() if k != 'derived_variables')):
                result += "\nNote: No data dictionary found. This may be because:\n"
                result += "1. The data path is incorrect\n"
                result += "2. The CSV files are not in the expected format\n"
                result += "3. The tables are empty\n"
                
            return [TextContent(type="text", text=result)]
            
        elif name == "explore_data":
            table_name = arguments.get("table_name")
            result = clif_server.data_explorer.explore_schema(table_name)
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "build_cohort":
            criteria = arguments["criteria"]
            save_cohort = arguments.get("save_cohort", False)
            
            cohort_df, cohort_id = clif_server.cohort_builder.build_cohort(criteria)
            current_cohort = cohort_df
            
            if save_cohort:
                clif_server.cohort_builder.save_cohort(cohort_id, cohort_df, criteria)
            
            # Get characteristics
            chars = clif_server.cohort_builder.get_cohort_characteristics(cohort_df)
            
            result = f"""Cohort built successfully!
Cohort ID: {cohort_id}
Size: {len(cohort_df)} hospitalizations
Unique patients: {cohort_df['patient_id'].nunique()}

Demographics:
- Mean age: {chars['demographics']['age_mean']:.1f} years
- Sex distribution: {chars['demographics']['sex_distribution']}

Clinical Characteristics:
- Mortality rate: {chars['clinical']['mortality_rate']:.1%}
- Mean ICU LOS: {chars['clinical']['mean_icu_los']:.1f} days
- Mechanical ventilation: {chars['clinical']['mechanical_ventilation_rate']:.1%}
"""
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "analyze_outcomes":
            cohort_id = arguments.get("cohort_id")
            outcomes = arguments["outcomes"]
            stratify_by = arguments.get("stratify_by", [])
            
            results = clif_server.outcomes_analyzer.analyze_outcomes(
                cohort_id, outcomes, stratify_by, []
            )
            
            formatted_results = clif_server.outcomes_analyzer.format_results(results)
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(formatted_results))]
            
        elif name == "compare_cohorts":
            cohort1_id = arguments["cohort1_id"]
            cohort2_id = arguments["cohort2_id"]
            outcomes = arguments["outcomes"]
            adjustment = arguments.get("adjustment_method", "unadjusted")
            
            results = clif_server.outcomes_analyzer.compare_cohorts(
                cohort1_id, cohort2_id, outcomes, adjustment
            )
            
            formatted_results = clif_server.outcomes_analyzer.format_comparison(results)
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(formatted_results))]
            
        elif name == "fit_regression":
            outcome = arguments["outcome"]
            predictors = arguments["predictors"]
            model_type = arguments["model_type"]
            interaction_terms = arguments.get("interaction_terms", [])
            random_effects = arguments.get("random_effects", [])
            
            # Use current cohort or load data
            if current_cohort is not None:
                df = current_cohort
            else:
                # Load and merge basic data
                patient_df = pd.read_csv(clif_server.data_path / "patient.csv")
                hosp_df = pd.read_csv(clif_server.data_path / "hospitalization.csv")
                df = patient_df.merge(hosp_df, on="patient_id")
            
            # Perform regression analysis
            from server.tools.regression_analyzer import RegressionAnalyzer
            regression = RegressionAnalyzer()
            
            result = regression.fit_model(
                df, outcome, predictors, model_type,
                interaction_terms, random_effects
            )
            
            # Store configuration for code generation
            last_regression_config = {
                'outcome': outcome,
                'predictors': predictors,
                'model_type': model_type,
                'interaction_terms': interaction_terms,
                'random_effects': random_effects
            }
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "descriptive_stats":
            variables = arguments["variables"]
            stratify_by = arguments.get("stratify_by")
            
            # Use current cohort or load data
            if current_cohort is not None:
                df = current_cohort
            else:
                patient_df = pd.read_csv(clif_server.data_path / "patient.csv")
                hosp_df = pd.read_csv(clif_server.data_path / "hospitalization.csv")
                df = patient_df.merge(hosp_df, on="patient_id")
            
            # Generate descriptive statistics
            from server.tools.stats_calculator import StatsCalculator
            stats_calc = StatsCalculator()
            
            result = stats_calc.calculate_descriptive_stats(df, variables, stratify_by)
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(result))]
            
        elif name == "build_ml_model":
            outcome = arguments["outcome"]
            features = arguments["features"]
            model_type = arguments.get("model_type", "auto")
            feature_selection = arguments.get("feature_selection")
            hyperparameter_tuning = arguments.get("hyperparameter_tuning", True)
            
            # Use current cohort or load data
            if current_cohort is not None:
                df = current_cohort
            else:
                patient_df = pd.read_csv(clif_server.data_path / "patient.csv")
                hosp_df = pd.read_csv(clif_server.data_path / "hospitalization.csv")
                df = patient_df.merge(hosp_df, on="patient_id")
                
                # Add derived outcomes if needed
                if outcome == "mortality" and "mortality" not in df.columns:
                    df["mortality"] = (df["discharge_disposition"] == "Expired").astype(int)
                if outcome == "los_days" and "los_days" not in df.columns:
                    df["admission_dttm"] = pd.to_datetime(df["admission_dttm"])
                    df["discharge_dttm"] = pd.to_datetime(df["discharge_dttm"])
                    df["los_days"] = (df["discharge_dttm"] - df["admission_dttm"]).dt.total_seconds() / 86400
            
            # Build model
            result = clif_server.ml_model_builder.build_model(
                df=df,
                outcome=outcome,
                features=features,
                model_type=model_type,
                feature_selection=feature_selection,
                hyperparameter_tuning=hyperparameter_tuning
            )
            
            # Generate report
            report = clif_server.ml_model_builder.generate_model_report(result['model_id'])
            
            # Store for code generation
            clif_server.code_generator.store_analysis({
                'type': 'ml_model',
                'model_info': result,
                'report': report
            })
            
            return [TextContent(type="text", text=clif_server.privacy_guard.sanitize_output(report))]
            
        elif name == "generate_code":
            language = arguments["language"]
            include_all = arguments.get("include_all", True)
            
            # Generate code based on stored analyses
            code = clif_server.code_generator.generate_code(
                "comprehensive_analysis", language, include_all, "script"
            )
            
            # Add regression if available
            if last_regression_config and include_all:
                regression_code = clif_server.code_generator.generate_regression_code(
                    last_regression_config, language
                )
                code += f"\n\n# Regression Analysis\n{regression_code}"
            
            # Save to file
            output_dir = Path.home() / "Desktop"
            output_dir.mkdir(exist_ok=True)
            
            filename = f"clif_analysis_{language}.{language[0]}"
            filepath = output_dir / filename
            
            with open(filepath, 'w') as f:
                f.write(code)
            
            result = f"Generated {language} code saved to: {filepath}\n\nPreview:\n{code[:500]}..."
            
            return [TextContent(type="text", text=result)]
            
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
            
    except Exception as e:
        error_msg = f"Error executing {name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return [TextContent(type="text", text=error_msg)]


async def main():
    """Main entry point"""
    global clif_server
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="CLIF MCP Server")
    parser.add_argument("--data-path", default="./data/synthetic", help="Path to CLIF CSV files")
    args = parser.parse_args()
    
    # Initialize server
    print(f"Initializing CLIF MCP Server with data path: {args.data_path}", file=sys.stderr)
    
    try:
        clif_server = CLIFServer(args.data_path)
        print("CLIF Server initialized successfully", file=sys.stderr)
        
    except Exception as e:
        print(f"Error initializing server: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())