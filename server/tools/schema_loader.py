"""
Dynamic schema loader for flexible database file discovery
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from glob import glob

logger = logging.getLogger(__name__)


class SchemaLoader:
    """Dynamically loads data files based on schema configuration"""
    
    def __init__(self, schema_path: str, data_path: str):
        """
        Initialize schema loader
        
        Args:
            schema_path: Path to schema configuration JSON file
            data_path: Base directory containing data files
        """
        self.data_path = Path(data_path)
        self.schema = self._load_schema(schema_path)
        self.discovered_files = {}
        self.loaded_tables = {}
        
    def _load_schema(self, schema_path: str) -> Dict[str, Any]:
        """Load schema configuration from JSON file"""
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            logger.info(f"Loaded schema: {schema.get('name', 'Unknown')} v{schema.get('version', 'Unknown')}")
            return schema
        except Exception as e:
            logger.error(f"Failed to load schema: {e}")
            raise
            
    def discover_files(self) -> Dict[str, List[Path]]:
        """
        Discover data files based on schema file patterns
        
        Returns:
            Dictionary mapping table names to list of matching file paths
        """
        self.discovered_files = {}
        
        for table_name, table_config in self.schema.get('tables', {}).items():
            file_pattern = table_config.get('file_pattern')
            if not file_pattern:
                logger.warning(f"No file pattern defined for table: {table_name}")
                continue
                
            # Search for files matching the pattern
            search_path = self.data_path / file_pattern
            matching_files = list(self.data_path.glob(file_pattern))
            
            if matching_files:
                self.discovered_files[table_name] = matching_files
                logger.info(f"Found {len(matching_files)} file(s) for table '{table_name}': {[f.name for f in matching_files]}")
            else:
                logger.warning(f"No files found for table '{table_name}' with pattern '{file_pattern}'")
                
        return self.discovered_files
        
    def load_table(self, table_name: str, required_columns: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Load a specific table based on schema
        
        Args:
            table_name: Name of the table to load
            required_columns: List of columns that must be present
            
        Returns:
            DataFrame with the table data, or None if not found
        """
        if table_name in self.loaded_tables:
            return self.loaded_tables[table_name]
            
        if table_name not in self.discovered_files:
            # Try to discover files if not already done
            self.discover_files()
            
        if table_name not in self.discovered_files:
            logger.error(f"Table '{table_name}' not found in discovered files")
            return None
            
        # Load the first matching file (or combine multiple files if needed)
        file_paths = self.discovered_files[table_name]
        
        try:
            if len(file_paths) == 1:
                df = pd.read_csv(file_paths[0])
            else:
                # Combine multiple files
                dfs = [pd.read_csv(fp) for fp in file_paths]
                df = pd.concat(dfs, ignore_index=True)
                
            # Validate required columns if specified
            if required_columns:
                missing_cols = set(required_columns) - set(df.columns)
                if missing_cols:
                    logger.warning(f"Missing required columns in {table_name}: {missing_cols}")
                    
            # Store in cache
            self.loaded_tables[table_name] = df
            logger.info(f"Loaded table '{table_name}' with {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load table '{table_name}': {e}")
            return None
            
    def get_table_info(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get schema information for a specific table"""
        return self.schema.get('tables', {}).get(table_name)
        
    def get_available_tables(self) -> List[str]:
        """Get list of all tables defined in schema"""
        return list(self.schema.get('tables', {}).keys())
        
    def get_table_columns(self, table_name: str) -> Dict[str, Any]:
        """Get column definitions for a table"""
        table_info = self.get_table_info(table_name)
        if table_info:
            return table_info.get('columns', {})
        return {}
        
    def validate_relationships(self) -> Dict[str, List[str]]:
        """
        Validate foreign key relationships between tables
        
        Returns:
            Dictionary of validation results
        """
        validation_results = {}
        
        for table_name, table_config in self.schema.get('tables', {}).items():
            foreign_keys = table_config.get('foreign_keys', {})
            if not foreign_keys:
                continue
                
            table_df = self.load_table(table_name)
            if table_df is None:
                validation_results[table_name] = [f"Table not found"]
                continue
                
            issues = []
            for fk_column, fk_reference in foreign_keys.items():
                ref_table, ref_column = fk_reference.split('.')
                ref_df = self.load_table(ref_table)
                
                if ref_df is None:
                    issues.append(f"Referenced table '{ref_table}' not found")
                    continue
                    
                if fk_column not in table_df.columns:
                    issues.append(f"Foreign key column '{fk_column}' not found")
                    continue
                    
                if ref_column not in ref_df.columns:
                    issues.append(f"Referenced column '{ref_column}' not found in '{ref_table}'")
                    continue
                    
                # Check for orphaned references
                orphaned = set(table_df[fk_column].dropna()) - set(ref_df[ref_column].dropna())
                if orphaned:
                    issues.append(f"Found {len(orphaned)} orphaned references in '{fk_column}'")
                    
            if issues:
                validation_results[table_name] = issues
                
        return validation_results
        
    def get_schema_summary(self) -> str:
        """Generate a human-readable summary of the schema"""
        summary_lines = [
            f"Schema: {self.schema.get('name', 'Unknown')} (v{self.schema.get('version', 'Unknown')})",
            f"Description: {self.schema.get('description', 'No description')}",
            "",
            "Available Tables:"
        ]
        
        for table_name in self.get_available_tables():
            table_info = self.get_table_info(table_name)
            if table_info:
                desc = table_info.get('description', 'No description')
                file_count = len(self.discovered_files.get(table_name, []))
                summary_lines.append(f"  - {table_name}: {desc} ({file_count} files found)")
                
        return "\n".join(summary_lines)