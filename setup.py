from setuptools import setup, find_packages

setup(
    name="ai-quality-framework",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pytest>=7.4.0",
        "pytest-xdist>=3.3.0",
        "pytest-html>=4.0.0",
        "openai>=1.0.0",
        "anthropic>=0.7.0",
        "langchain>=0.1.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.17.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "loguru>=0.7.0",
        "requests>=2.31.0",
        "tabulate>=0.9.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.6",
        "aiohttp>=3.9.0",
        "tenacity>=8.2.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-quality=main:cli",
        ],
    },
)
