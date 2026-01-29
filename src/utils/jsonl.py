from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import json

def append_jsonl(path: str, record: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
