from pathlib import Path
import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]


def load_yaml(filename: str) -> dict:
    """Load a YAML configuration from the configs directory."""

    config_path = ROOT_DIR / "configs" / filename

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_root_dir() -> Path:
    """Return the project root directory."""
    return ROOT_DIR