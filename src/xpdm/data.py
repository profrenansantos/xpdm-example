"""Load the NASA C-MAPSS turbofan degradation data (Saxena et al., 2008).

The data are not redistributed in this repository. `download()` fetches the
official NASA archive; if that fails you can pass a mirror URL or place the
text files in `data_dir` yourself.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

NASA_URL = ("https://phm-datasets.s3.amazonaws.com/NASA/"
            "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip")

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]

# Rows per file, used as a sanity check after download.
EXPECTED_ROWS = {"train_FD001": 20631, "test_FD001": 13096, "RUL_FD001": 100}


def _extract_nested(blob: bytes, target: Path) -> None:
    """Extract every C-MAPSS text file from a zip, descending into nested zips."""
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        for name in zf.namelist():
            base = Path(name).name
            if base.lower().endswith(".zip"):
                _extract_nested(zf.read(name), target)
            elif base.startswith(("train_FD", "test_FD", "RUL_FD")) and base.endswith(".txt"):
                (target / base).write_bytes(zf.read(name))


def download(data_dir: str | Path = "data", url: str = NASA_URL,
             mirror: str | None = None, subset: str = "FD001") -> Path:
    """Download C-MAPSS into `data_dir`. `mirror` is a base URL holding the raw .txt files."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if all((data_dir / f"{p}_{subset}.txt").exists() for p in ("train", "test", "RUL")):
        return data_dir
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            _extract_nested(r.read(), data_dir)
    except Exception as err:  # network or format problem
        if not mirror:
            raise RuntimeError(
                f"Could not download {url} ({err}). Pass --mirror, or put "
                f"train_{subset}.txt, test_{subset}.txt and RUL_{subset}.txt in {data_dir}.") from err
        for p in ("train", "test", "RUL"):
            urllib.request.urlretrieve(f"{mirror.rstrip('/')}/{p}_{subset}.txt", data_dir / f"{p}_{subset}.txt")
    return data_dir


def load_cmapss(subset: str = "FD001", data_dir: str | Path = "data"):
    """Return (train, test, rul_test). `rul_test` holds the true RUL at the last test cycle."""
    data_dir = Path(data_dir)
    read = lambda f: pd.read_csv(data_dir / f, sep=r"\s+", header=None)
    train = read(f"train_{subset}.txt").iloc[:, :26]
    test = read(f"test_{subset}.txt").iloc[:, :26]
    train.columns = COLUMNS
    test.columns = COLUMNS
    rul = read(f"RUL_{subset}.txt").iloc[:, 0].rename("rul_true")
    rul.index = pd.RangeIndex(1, len(rul) + 1, name="unit")
    for name, df in (("train", train), ("test", test), ("RUL", rul)):
        exp = EXPECTED_ROWS.get(f"{name}_{subset}")
        if exp and len(df) != exp:
            raise ValueError(f"{name}_{subset}: expected {exp} rows, found {len(df)}")
    return train, test, rul


def add_rul(train: pd.DataFrame, clip: int | None = 125) -> pd.DataFrame:
    """Add remaining useful life to run-to-failure data, optionally clipped (piecewise-linear target)."""
    out = train.copy()
    last = out.groupby("unit")["cycle"].transform("max")
    out["rul"] = last - out["cycle"]
    if clip is not None:
        out["rul"] = out["rul"].clip(upper=clip)
    return out
