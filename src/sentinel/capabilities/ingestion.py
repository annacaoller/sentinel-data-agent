
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd



def _project_root() -> Path:
    # ingestion.py -> capabilities -> sentinel -> src -> <repo root>
    return Path(__file__).resolve().parents[3]


def validate_dataset_path(dataset_path: str | Path) -> Path:
    """
    Resolve dataset_path aceitando:
    - caminho absoluto
    - caminho relativo ao cwd
    - caminho relativo à raiz do repo
    - caminho relativo a <repo>/examples
    """
    p = Path(dataset_path).expanduser()

    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        root = _project_root()
        candidates.append(root / p)
        candidates.append(root / "examples" / p)

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()

    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Dataset not found. Tried:\n{tried}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # se você já tem lógica por extensão, mantenha; aqui é um exemplo mínimo
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def profile_dataset(dataset_path: str | Path, sample_size: int = 10) -> dict[str, Any]:
    """
    Se você já tem um DatasetProfile pydantic/dataclass, troque o retorno para ele.
    Aqui eu deixei como dict para ficar claro o shape e não depender do seu schema.
    """
    path = validate_dataset_path(dataset_path)
    df = load_dataset(path)

    safe_sample_size = max(0, min(sample_size, len(df)))

    # Normaliza chaves para str para satisfazer typing e evitar Pylance "Hashable"
    raw_rows: list[dict[Any, Any]] = df.head(safe_sample_size).to_dict(orient="records")
    sample_rows: list[dict[str, Any]] = [
        {str(k): v for k, v in row.items()} for row in raw_rows
    ]

    raw_dtypes: dict[Any, str] = df.dtypes.astype(str).to_dict()
    dtypes: dict[str, str] = {str(k): str(v) for k, v in raw_dtypes.items()}

    columns: list[str] = [str(c) for c in df.columns]

    return {
        "dataset_path": str(path),
        "file_type": path.suffix.lower().lstrip("."),
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": columns,
        "dtypes": dtypes,
        "sample_rows": sample_rows,
    }

