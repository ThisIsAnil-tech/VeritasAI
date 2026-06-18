import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Resolve workspace/project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class ProjectConfig(BaseModel):
    name: str = "AI Quality Evaluation Framework"
    version: str = "1.0.0"

class LLMModelConfig(BaseModel):
    name: str
    provider: str
    temperature: float = 0.7
    max_tokens: int = 2000

class LLMConfig(BaseModel):
    default_model: str = "gpt-4"
    models: List[LLMModelConfig] = []

class MetricThresholds(BaseModel):
    threshold: float
    method: str = "cosine"
    runs: Optional[int] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    use_llm_judge: Optional[bool] = False

class TestingConfig(BaseModel):
    parallel_workers: int = 4
    timeout: int = 30
    retry_attempts: int = 3

class ReportingConfig(BaseModel):
    output_dir: str = "./reports"
    formats: List[str] = ["html", "json"]
    include_charts: bool = True
    include_raw_data: bool = True

class DatasetsConfig(BaseModel):
    golden_path: str = "./src/data/datasets/golden_dataset.json"
    edge_path: str = "./src/data/datasets/edge_cases.json"
    domain_path: str = "./src/data/datasets/domain_specific.json"

class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    metrics: Dict[str, Any] = {}
    testing: TestingConfig = Field(default_factory=TestingConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    datasets: DatasetsConfig = Field(default_factory=DatasetsConfig)
    
    # Environment keys injected automatically
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_api_base: Optional[str] = None

def get_config_path(filename: str) -> Path:
    """Return the absolute path of a config file under the config/ folder."""
    # Look for config files first in the project config directory
    path = PROJECT_ROOT / "config" / filename
    if path.exists():
        return path
    # Fallback to local execution directory
    fallback = Path("config") / filename
    if fallback.exists():
        return fallback.resolve()
    return path

def load_yaml(path: Path) -> Dict[str, Any]:
    """Helper to load yaml files safely."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_app_config() -> AppConfig:
    """Load configuration from files, merging yaml settings with environment variables."""
    main_config_path = get_config_path("config.yaml")
    models_config_path = get_config_path("models.yaml")
    metrics_config_path = get_config_path("metrics_config.yaml")

    config_data = load_yaml(main_config_path)
    models_data = load_yaml(models_config_path)
    metrics_data = load_yaml(metrics_config_path)

    # Merge models list
    if "models" in models_data and "llm" in config_data:
        # Build model configs from models.yaml if present
        model_list = []
        for model_name, details in models_data["models"].items():
            model_list.append({
                "name": model_name,
                "provider": details.get("provider", "mock"),
                "temperature": details.get("temperature", 0.7),
                "max_tokens": details.get("max_tokens", 2000)
            })
        config_data["llm"]["models"] = model_list

    # Merge metrics settings
    if "metrics" in metrics_data:
        if "metrics" not in config_data:
            config_data["metrics"] = {}
        config_data["metrics"].update(metrics_data["metrics"])

    # Load from environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1")

    # Instantiate AppConfig
    app_config = AppConfig(**config_data)
    app_config.openai_api_key = openai_key
    app_config.anthropic_api_key = anthropic_key
    app_config.ollama_api_base = ollama_base

    return app_config

# Singleton instance
config_instance = get_app_config()
