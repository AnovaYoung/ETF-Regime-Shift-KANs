from pathlib import Path
import pandas as pd

def discover_fundflow_dirs(root: str | Path) -> list[Path]:
    """Return a list of ticker directories under data/raw/fundflow."""
    root = Path(root)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])

def load_features_parquet(path: str | Path) -> pd.DataFrame:
    """Load a parquet of engineered features."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)
