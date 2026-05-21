from setuptools import setup, find_packages

setup(
    name="sodil",
    version="1.0.0",
    description="Systematic Opportunity Discovery & Investment Lab",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "yfinance>=0.2.40",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "plotly>=5.18.0",
        "rich>=13.7.0",
        "typer>=0.9.0",
        "requests>=2.31.0",
        "SQLAlchemy>=2.0.0",
        "python-dateutil>=2.8.2",
        "pytz>=2024.1",
        "tqdm>=4.66.0",
        "tabulate>=0.9.0",
        "colorama>=0.4.6",
    ],
    entry_points={
        "console_scripts": [
            "sodil=src.cli.main:main",
        ],
    },
)
