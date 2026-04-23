from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "store_intel.db"))
STORE_LAYOUT_PATH = Path(
    os.getenv("STORE_LAYOUT_PATH", DATA_DIR / "sample" / "store_layout.json")
)

