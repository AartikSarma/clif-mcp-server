# Claude Desktop Setup Guide for CLIF MCP Server

## Prerequisites

1. Install Claude Desktop from: https://claude.ai/download
2. Ensure Python 3.8+ is installed
3. Complete the CLIF MCP Server setup (synthetic data generated)

## Configuration Steps

### 1. Locate your Claude Desktop config file

The configuration file is located at:
- **macOS**: `~/.claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`

### 2. Add the CLIF MCP Server

Edit the config file and add the CLIF server configuration:

```json
{
  "mcpServers": {
    "clif": {
      "command": "python",
      "args": ["-m", "server.main", "--data-path", "./data/synthetic"],
      "cwd": "."
    }
  }
}
```

**Important**: Update the paths to match your system:
- The `cwd` should be set to the absolute path of your clif-mcp-server directory
- Ensure the data path points to your synthetic (or real) CLIF data

### 3. Install Dependencies

Make sure all required packages are installed:

```bash
cd /path/to/clif-mcp-server
pip install -r requirements.txt
```

### 4. Restart Claude Desktop

After saving the configuration:
1. Quit Claude Desktop completely
2. Restart Claude Desktop
3. The CLIF tools should now be available

## Testing the Integration

Once configured, you can test with these queries in Claude Desktop:

### Basic Data Exploration
```
"Explore the CLIF schema and show me what tables are available"
```

### Cohort Building
```
"Build a cohort of mechanically ventilated patients over 65 years old"
```

### Outcomes Analysis
```
"Analyze ICU mortality for patients who received vasopressors"
```

### Comparative Analysis
```
"Compare outcomes between patients with short (<3 days) vs long (>7 days) ICU stays"
```

### Code Generation
```
"Generate Python code for analyzing 30-day readmission rates"
```

## Troubleshooting

### Server doesn't start
- Check Python path: `which python` or `where python`
- Verify all dependencies installed: `pip list`
- Check logs in Claude Desktop developer console

### Tools not available
- Ensure configuration file is valid JSON
- Check file paths are absolute, not relative
- Verify data directory exists and contains CSV files

### Permission errors
- Ensure read permissions on data directory
- Check write permissions for output directories

## Using Real CLIF Data

To use with real CLIF data instead of synthetic:

1. Update the data path in config:
```json
"args": ["-m", "server.main", "--data-path", "/path/to/real/clif/data"]
```

2. Ensure your CLIF data follows the expected format:
- CSV files for each table
- Standard CLIF 2.1.0 column names
- UTF-8 encoding

## Privacy Considerations

The MCP server includes privacy protections:
- Small cell suppression (<5 counts)
- No patient identifiers in outputs
- Local data only - no external transmission
- Audit logging of all queries

## Advanced Configuration

### Custom Python Environment
If using conda or virtualenv:

```json
{
  "mcpServers": {
    "clif": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "server.main", "--data-path", "./data"],
      "cwd": "/absolute/path/to/clif-mcp-server",
      "env": {
        "PYTHONPATH": "/absolute/path/to/clif-mcp-server"
      }
    }
  }
}
```

### Multiple Data Sources
You can configure multiple instances for different datasets:

```json
{
  "mcpServers": {
    "clif-synthetic": {
      "command": "python",
      "args": ["-m", "server.main", "--data-path", "./data/synthetic"],
      "cwd": "/absolute/path/to/clif-mcp-server"
    },
    "clif-production": {
      "command": "python",
      "args": ["-m", "server.main", "--data-path", "/secure/clif/data"],
      "cwd": "/absolute/path/to/clif-mcp-server"
    }
  }
}
```

## Support

For issues or questions:
- Check the main README.md
- Review server logs
- Ensure CLIF data format compliance