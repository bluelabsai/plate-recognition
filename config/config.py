
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
STORE_DIR = Path(BASE_DIR, "store")
MODELS_DIR = Path(STORE_DIR, "models")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)

# hard rule
TRANSFORM_LASTCHAR = {
    "O": "D",
    "0": "D"
}