from pathlib import Path

def resolve_path(p: str | Path) -> Path:
    return Path(p).expanduser().resolve()
