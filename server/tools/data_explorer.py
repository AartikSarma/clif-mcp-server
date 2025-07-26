"""
Dynamic Data Explorer Tool for CLIF MCP Server
Provides schema exploration and data quality metrics using dynamic schema
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import logging

from .schema_loader import SchemaLoader

logger = logging.getLogger(__name__)


class DataExplorer:
    def __init__(self, schema_path: str, data_path: str):
        """
        Initialize data explorer with dynamic schema
        
        Args:
            schema_path: Path to schema configuration JSON
            data_path: Base directory containing data files
        """
        self.data_path = Path(data_path)
        self.schema_loader = SchemaLoader(schema_path, data_path)
        
        # Discover available files
        self.available_files = self.schema_loader.discover_files()
        
    def explore_schema(self, table_name: Optional[str] = None) -> str:
        """Explore dataset schema and quality metrics"""
        
        if table_name:
            return self._explore_single_table(table_name)
        else:
            return self._explore_all_tables()
    
    def _explore_all_tables(self) -> str:
        """Provide overview of all tables"""
        output = []
        output.append("Dataset Overview (Dynamic Schema)")
        output.append("=" * 50)
        output.append(self.schema_loader.get_schema_summary())
        output.append("")
        output.append("Discovered Files:")
        output.append("-" * 30)
        
        for table_name, file_paths in self.available_files.items():
            output.append(f"\n{table_name.upper()}:")
            
            # Get table info from schema
            table_info = self.schema_loader.get_table_info(table_name)
            if table_info:
                output.append(f"  Description: {table_info.get('description', 'No description')}")
                output.append(f"  Files found: {len(file_paths)}")
                for fp in file_paths:
                    output.append(f"    - {fp.name}")
                    
            # Try to get basic stats
            try:
                df_sample = self.schema_loader.load_table(table_name)
                if df_sample is not None:
                    output.append(f"  Rows: {len(df_sample):,}")
                    output.append(f"  Columns: {len(df_sample.columns)}")
                    
                    # Show column info from schema
                    columns_info = table_info.get('columns', {}) if table_info else {}
                    if columns_info:
                        output.append("  Schema columns:")
                        for col_name, col_info in list(columns_info.items())[:5]:
                            col_type = col_info.get('type', 'unknown')
                            required = "required" if col_info.get('required', False) else "optional"
                            output.append(f"    - {col_name} ({col_type}, {required})")
                        if len(columns_info) > 5:
                            output.append(f"    ... and {len(columns_info) - 5} more columns")
                            
            except Exception as e:
                output.append(f"  Error loading table: {str(e)}")
        
        # Show tables defined in schema but not found
        schema_tables = set(self.schema_loader.get_available_tables())
        found_tables = set(self.available_files.keys())
        missing_tables = schema_tables - found_tables
        
        if missing_tables:
            output.append("\nTables defined in schema but not found:")
            for table in missing_tables:
                output.append(f"  - {table}")
                
        return "\n".join(output)
    
    def _explore_single_table(self, table_name: str) -> str:
        """Detailed exploration of a single table"""
        
        available_tables = self.schema_loader.get_available_tables()
        if table_name not in available_tables:
            return f"Table '{table_name}' not found in schema. Available tables: {', '.join(available_tables)}"
        
        # Get table info from schema
        table_info = self.schema_loader.get_table_info(table_name)
        
        output = []
        output.append(f"Table: {table_name.upper()}")
        output.append("=" * 50)
        
        if table_info:
            output.append(f"Description: {table_info.get('description', 'No description')}")
            if 'primary_key' in table_info:
                output.append(f"Primary Key: {table_info['primary_key']}")
            if 'foreign_keys' in table_info:
                output.append("Foreign Keys:")
                for fk, ref in table_info['foreign_keys'].items():
                    output.append(f"  - {fk} -> {ref}")
        
        # Check if files exist
        if table_name not in self.available_files:
            output.append("\nStatus: No files found matching pattern")
            if table_info and 'file_pattern' in table_info:
                output.append(f"Expected pattern: {table_info['file_pattern']}")
            return "\n".join(output)
        
        # Load and analyze the table
        try:
            df = self.schema_loader.load_table(table_name)
            if df is None:
                output.append("\nStatus: Could not load table")
                return "\n".join(output)
                
            output.append(f"\nData Statistics:")
            output.append(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Column analysis
            output.append("\nColumn Analysis:")
            output.append("-" * 80)
            output.append(f"{'Column':<30} {'Type':<15} {'Non-Null':<10} {'Unique':<10} {'Schema Type':<15}")
            output.append("-" * 80)
            
            columns_info = table_info.get('columns', {}) if table_info else {}
            
            for col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                unique = df[col].nunique()
                schema_type = columns_info.get(col, {}).get('type', 'not in schema')
                
                output.append(f"{col:<30} {dtype:<15} {non_null:<10,} {unique:<10,} {schema_type:<15}")
            
            # Show columns in schema but not in data
            schema_cols = set(columns_info.keys())
            data_cols = set(df.columns)
            missing_cols = schema_cols - data_cols
            
            if missing_cols:
                output.append("\nColumns defined in schema but not found in data:")
                for col in missing_cols:
                    col_info = columns_info[col]
                    required = "required" if col_info.get('required', False) else "optional"
                    output.append(f"  - {col} ({col_info.get('type', 'unknown')}, {required})")
            
            # Data quality metrics
            output.append("\nData Quality Metrics:")
            output.append(f"Total missing values: {df.isna().sum().sum():,}")
            output.append(f"Columns with missing data: {(df.isna().sum() > 0).sum()} / {len(df.columns)}")
            
            # Sample data
            output.append("\nSample Data (first 5 rows):")
            sample_df = df.head()
            output.append(self._format_dataframe_sample(sample_df))
            
            # Validate relationships if this table has foreign keys
            if table_info and 'foreign_keys' in table_info:
                output.append("\nRelationship Validation:")
                validation_results = self._validate_table_relationships(table_name, df)
                for result in validation_results:
                    output.append(f"  - {result}")
                    
        except Exception as e:
            output.append(f"\nError analyzing table: {str(e)}")
            logger.error(f"Error exploring table {table_name}: {e}")
        
        return "\n".join(output)
    
    def _format_dataframe_sample(self, df: pd.DataFrame) -> str:
        """Format dataframe sample for display"""
        # Limit column width for display
        with pd.option_context('display.max_columns', None, 
                             'display.width', 120,
                             'display.max_colwidth', 20):
            return str(df)
    
    def _validate_table_relationships(self, table_name: str, df: pd.DataFrame) -> List[str]:
        """Validate foreign key relationships for a table"""
        results = []
        table_info = self.schema_loader.get_table_info(table_name)
        
        if not table_info or 'foreign_keys' not in table_info:
            return results
            
        for fk_column, fk_reference in table_info['foreign_keys'].items():
            if fk_column not in df.columns:
                results.append(f"Foreign key column '{fk_column}' not found in data")
                continue
                
            ref_table, ref_column = fk_reference.split('.')
            ref_df = self.schema_loader.load_table(ref_table)
            
            if ref_df is None:
                results.append(f"Referenced table '{ref_table}' not found")
                continue
                
            if ref_column not in ref_df.columns:
                results.append(f"Referenced column '{ref_column}' not found in '{ref_table}'")
                continue
                
            # Check for orphaned references
            fk_values = set(df[fk_column].dropna().unique())
            ref_values = set(ref_df[ref_column].dropna().unique())
            orphaned = fk_values - ref_values
            
            if orphaned:
                results.append(f"{fk_column} -> {fk_reference}: {len(orphaned)} orphaned references")
            else:
                results.append(f"{fk_column} -> {fk_reference}: All references valid ✓")
                
        return results
    
    def get_column_values(self, table_name: str, column_name: str, limit: int = 20) -> Dict[str, Any]:
        """Get unique values and statistics for a specific column"""
        df = self.schema_loader.load_table(table_name)
        if df is None or column_name not in df.columns:
            return {"error": f"Table '{table_name}' or column '{column_name}' not found"}
            
        col_data = df[column_name]
        
        result = {
            "table": table_name,
            "column": column_name,
            "total_rows": len(df),
            "non_null_count": col_data.notna().sum(),
            "null_count": col_data.isna().sum(),
            "unique_count": col_data.nunique(),
            "dtype": str(col_data.dtype)
        }
        
        # Get value distribution
        if col_data.dtype in ['object', 'string']:
            value_counts = col_data.value_counts().head(limit)
            result["top_values"] = value_counts.to_dict()
        elif np.issubdtype(col_data.dtype, np.number):
            result["statistics"] = {
                "min": float(col_data.min()) if not col_data.empty else None,
                "max": float(col_data.max()) if not col_data.empty else None,
                "mean": float(col_data.mean()) if not col_data.empty else None,
                "median": float(col_data.median()) if not col_data.empty else None,
                "std": float(col_data.std()) if not col_data.empty else None
            }
            
        return result
    
    def validate_schema(self) -> Dict[str, Any]:
        """Validate the entire schema against the actual data"""
        validation_report = {
            "schema_name": self.schema_loader.schema.get('name', 'Unknown'),
            "schema_version": self.schema_loader.schema.get('version', 'Unknown'),
            "tables": {},
            "overall_status": "valid"
        }
        
        for table_name in self.schema_loader.get_available_tables():
            table_validation = {
                "status": "valid",
                "issues": [],
                "file_found": table_name in self.available_files
            }
            
            if not table_validation["file_found"]:
                table_validation["status"] = "missing"
                table_validation["issues"].append("No files found matching pattern")
            else:
                # Validate table structure
                df = self.schema_loader.load_table(table_name)
                if df is not None:
                    table_info = self.schema_loader.get_table_info(table_name)
                    columns_info = table_info.get('columns', {}) if table_info else {}
                    
                    # Check required columns
                    for col_name, col_info in columns_info.items():
                        if col_info.get('required', False) and col_name not in df.columns:
                            table_validation["status"] = "invalid"
                            table_validation["issues"].append(f"Required column '{col_name}' missing")
                            
                    # Check data types (basic validation)
                    for col_name in df.columns:
                        if col_name in columns_info:
                            expected_type = columns_info[col_name].get('type')
                            if expected_type == 'datetime':
                                try:
                                    pd.to_datetime(df[col_name], errors='coerce')
                                except:
                                    table_validation["issues"].append(f"Column '{col_name}' cannot be parsed as datetime")
                                    
            validation_report["tables"][table_name] = table_validation
            
            if table_validation["status"] != "valid":
                validation_report["overall_status"] = "invalid"
                
        return validation_report