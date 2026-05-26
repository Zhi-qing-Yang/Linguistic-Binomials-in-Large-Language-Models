import json
from pathlib import Path

import pandas as pd


def load_jsonl(path):
    path = Path(path)
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)


def load_many_jsonl(paths):
    dfs = [load_jsonl(path) for path in paths]
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
