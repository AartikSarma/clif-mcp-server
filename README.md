# CLIF MCP Server

A Model Context Protocol (MCP) server for AI-assisted clinical research using flexible database schemas. Supports the Common Longitudinal ICU Format (CLIF) and any custom clinical database structure through configuration files.

## Overview

This MCP server enables researchers to:
- Build complex patient cohorts using clinical criteria
- Analyze clinical outcomes (mortality, length of stay, complications)
- Generate reproducible analysis code for multi-site validation
- Ensure data privacy with built-in security measures

## Features

### Clinical Research Tools
- **Cohort Builder**: Create patient cohorts with complex inclusion/exclusion criteria
- **Outcomes Analysis**: Analyze mortality, ICU/hospital LOS, ventilation, readmissions
- **Statistical Comparisons**: Compare outcomes between cohorts with various adjustment methods
- **Code Generation**: Generate Python/R scripts for reproducible analyses

### Privacy & Security
- All data remains local - no external transmission
- Cell suppression for small counts
- Removal of patient identifiers from outputs
- Audit logging of all operations
- Differential privacy options

## Installation

1. Clone the repository:
```bash
cd clif-mcp-server
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your schema configuration:
   - Use the provided `schema_config.json` for CLIF format data
   - Or create a custom schema file for your database structure (see `examples/custom_schema_example.json`)

4. (Optional) Generate synthetic test data:
```bash
python synthetic_data/clif_generator.py
```

## Usage

### Starting the MCP Server

```bash
python clif_server.py --schema-path schema_config.json --data-path ./data/synthetic
```

For custom schemas:
```bash
python clif_server.py --schema-path your_schema.json --data-path ./your_data
```

### Using with Claude Desktop

1. Add the server to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "clif": {
      "command": "python",
      "args": [
        "clif_server.py",
        "--schema-path", "schema_config.json",
        "--data-path", "/path/to/your/data"
      ],
      "cwd": "/path/to/clif-mcp-server"
    }
  }
}
```

2. Start Claude Desktop and use the CLIF tools for analysis

### Example Analyses

#### First, explore your data
```
Show me the available tables and their structure
```

#### Validate the schema
```
Validate that the schema matches the actual data files
```

#### Building a Cohort
```
Build a cohort of mechanically ventilated patients aged 18-65 with ICU stays > 3 days
```

#### Using Custom Filters
```
Build a cohort with custom filters: patients with creatinine > 2.0 in the labs table
```

#### Analyzing Outcomes
```
Analyze mortality and ICU length of stay for the ventilated cohort, stratified by age groups
```

#### Generating Code
```
Generate Python code for the mortality analysis that can be run at other sites
```

## Schema Configuration

The server uses a JSON schema file to understand your database structure. This allows it to work with any clinical database format.

### Schema Format

```json
{
  "name": "Your Schema Name",
  "version": "1.0.0",
  "tables": {
    "table_name": {
      "description": "What this table contains",
      "file_pattern": "filename_pattern*.csv",
      "primary_key": "id_column",
      "foreign_keys": {
        "fk_column": "referenced_table.column"
      },
      "columns": {
        "column_name": {
          "type": "string|integer|float|datetime",
          "required": true,
          "description": "Column description"
        }
      }
    }
  }
}
```

### Default CLIF Schema

The provided `schema_config.json` supports CLIF 2.1.0 format with tables for:
- Patient demographics
- Hospitalizations
- ADT (transfers)
- Vital signs
- Laboratory results
- Respiratory support
- Medications
- Clinical assessments
- Patient positioning

### Custom Schemas

Create your own schema file to work with different database structures. See `examples/custom_schema_example.json` for a template.

## Multi-Site Validation

Generated analysis scripts can be shared with other CLIF sites for validation:

1. Share the generated script (no patient data)
2. Each site runs the script on their local data
3. Sites share aggregate results (summary statistics only)
4. Compare results across sites to validate findings

## Development

### Adding New Outcomes

To add new outcome measures, modify `server/tools/outcomes_analyzer.py`:

```python
def _analyze_new_outcome(self, cohort: pd.DataFrame) -> Dict[str, Any]:
    # Implement outcome calculation
    pass
```

### Adding New Cohort Criteria

To add new selection criteria, modify `server/tools/cohort_builder.py`:

```python
# In build_cohort method
if 'new_criteria' in criteria:
    # Apply new filter
    pass
```

## Security Considerations

- Never commit real patient data
- Review generated code before sharing
- Use aggregate statistics only for multi-site comparisons
- Enable audit logging in production
- Consider differential privacy for sensitive analyses

## Contributing

Please submit issues and pull requests to improve the server.

## License

This project is licensed under the MIT License.