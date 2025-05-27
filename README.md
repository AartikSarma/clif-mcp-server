# CLIF MCP Server

A Model Context Protocol (MCP) server for AI-assisted clinical research using the Common Longitudinal ICU Format (CLIF) datasets.

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

3. Generate synthetic test data:
```bash
python synthetic_data/clif_generator.py
```

## Usage

### Starting the MCP Server

```bash
python -m server.main --data-path ./data/synthetic
```

### Using with Claude Desktop

1. Add the server to your Claude Desktop configuration:
```json
{
  "mcpServers": {
    "clif": {
      "command": "python",
      "args": ["-m", "server.main", "--data-path", "/path/to/your/clif/data"],
      "cwd": "/path/to/clif-mcp-server"
    }
  }
}
```

2. Start Claude Desktop and use the CLIF tools for analysis

### Example Analyses

#### Building a Cohort
```
Build a cohort of mechanically ventilated patients aged 18-65 with ICU stays > 3 days
```

#### Analyzing Outcomes
```
Analyze mortality and ICU length of stay for the ventilated cohort, stratified by age groups
```

#### Comparing Cohorts
```
Compare outcomes between patients who received early vs late mechanical ventilation
```

#### Generating Code
```
Generate Python code for the mortality analysis that can be run at other CLIF sites
```

## Data Format

The server expects CLIF 2.1.0 format CSV files:
- `patient.csv`: Demographics
- `hospitalization.csv`: Admission/discharge info
- `adt.csv`: Location transfers
- `vitals.csv`: Vital signs
- `labs.csv`: Laboratory results
- `respiratory_support.csv`: Ventilation data
- `medication_administration.csv`: Medications
- `patient_assessments.csv`: Clinical assessments
- `position.csv`: Patient positioning

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