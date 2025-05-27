"""
Setup script for CLIF MCP Server
"""

from setuptools import setup, find_packages

setup(
    name="clif-mcp-server",
    version="0.1.0",
    description="Model Context Protocol server for AI-assisted clinical research with CLIF datasets",
    author="CLIF Consortium",
    packages=find_packages(),
    install_requires=[
        "mcp>=0.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "lifelines>=0.27.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jinja2>=3.1.0",
        "faker>=18.0.0",
        "pydantic>=2.0"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "clif-mcp-server=server.main:main",
            "clif-generate-data=synthetic_data.clif_generator:main"
        ]
    }
)