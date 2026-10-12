"""Lightweight process mining: directly-follows graph, variants and conformance checking.

Implemented directly in pandas so the package stays MIT-licensed; pm4py (AGPL)
can be used on the same log via `eventlog.to_xes_columns` if you prefer.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd

# Reference model of the intended maintenance process: allowed directly-follows pairs.
REFERENCE = {
    ("START", "Monitoring started"),
    ("Monitoring started", "Degradation alert"),
    ("Degradation alert", "Inspection scheduled"),
    ("Inspection scheduled", "Inspection performed"),
    ("Inspection performed", "Engine removal (planned)"),
    ("Engine removal (planned)", "END"),
}


def _traces(log: pd.DataFrame) -> pd.Series:
    return log.sort_values(["case_id", "cycle"]).groupby("case_id")["activity"].apply(list)


def dfg(log: pd.DataFrame) -> pd.DataFrame:
    """Directly-follows counts and mean waiting time (cycles) between activities."""
    log = log.sort_values(["case_id", "cycle"])
    nxt = log.groupby("case_id")[["activity", "cycle"]].shift(-1)
    pairs = pd.DataFrame({"source": log["activity"], "target": nxt["activity"],
                          "wait": nxt["cycle"] - log["cycle"]}).dropna()
    out = pairs.groupby(["source", "target"]).agg(count=("wait", "size"), mean_wait=("wait", "mean"))
    starts = log.groupby("case_id")["activity"].first().value_counts()
    ends = log.groupby("case_id")["activity"].last().value_counts()
    extra = [pd.DataFrame({"source": "START", "target": starts.index, "count": starts.values, "mean_wait": 0.0}),
             pd.DataFrame({"source": ends.index, "target": "END", "count": ends.values, "mean_wait": 0.0})]
    return pd.concat([out.reset_index(), *extra], ignore_index=True)


def variants(log: pd.DataFrame) -> pd.DataFrame:
    v = _traces(log).apply(" → ".join).value_counts()
    return v.rename_axis("variant").reset_index(name="cases")


def conformance(log: pd.DataFrame, reference: set = REFERENCE) -> pd.DataFrame:
    """Per case: share of directly-follows steps allowed by the reference model, and the deviations."""
    rows = []
    for case, trace in _traces(log).items():
        steps = list(zip(["START", *trace], [*trace, "END"]))
        bad = [f"{a} → {b}" for a, b in steps if (a, b) not in reference]
        rows.append({"case_id": case, "fitness": 1 - len(bad) / len(steps),
                     "conforms": not bad, "deviations": "; ".join(bad)})
    return pd.DataFrame(rows)


def render_dfg(d: pd.DataFrame, path: str | Path) -> Path | None:
    """Write a Graphviz .dot file and, if `dot` is installed, a PNG next to it."""
    path = Path(path)
    lines = ["digraph DFG {", "rankdir=TB; node [shape=box, style=rounded, fontname=Helvetica];"]
    maxc = d["count"].max()
    for r in d.itertuples():
        lab = f"{r.count}" + (f"\\n{r.mean_wait:.1f} cyc" if r.mean_wait else "")
        colour = "firebrick" if "failure" in f"{r.source}{r.target}".lower() else "gray30"
        lines.append(f'"{r.source}" -> "{r.target}" [label="{lab}", penwidth={1 + 4 * r.count / maxc:.2f}, color={colour}];')
    lines.append("}")
    dot_path = path.with_suffix(".dot")
    dot_path.write_text("\n".join(lines))
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(path.with_suffix(".png"))], check=True)
        return path.with_suffix(".png")
    return None
