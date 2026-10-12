"""End-to-end pipeline: RUL model -> simulated event log and process mining -> MCDA ranking -> explanations.

    python -m xpdm.run --subset FD001 --data-dir data --out outputs [--mirror URL] [--llm]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import eventlog, explain, mcda, model as model_mod, processmining
from .data import add_rul, download, load_cmapss
from .features import (build_features, degradation_rate, feature_columns, health_index,
                       informative_sensors, last_cycle)
from .model import RULModel, out_of_fold, phm_score, rmse

CLIP = 125
_T0 = None


def log(msg: str) -> None:
    """Progress message with elapsed time; flushed so it shows immediately on Windows terminals too."""
    import time
    global _T0
    _T0 = _T0 or time.perf_counter()
    print(f"[{time.perf_counter() - _T0:6.1f}s] {msg}", flush=True)

# Criteria in priority order (used for rank-order-centroid weights). False = lower value is more urgent.
CRITERIA = {"rul_lower": False, "degradation_rate": True, "process_deviations": True}


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        fmt = lambda v: (str(int(v)) if float(v).is_integer() else f"{v:.3f}") if isinstance(v, (int, float, np.number)) else str(v)
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    return "\n".join(lines)


def run(subset="FD001", data_dir="data", out="outputs", mirror=None, use_llm=False, top=10,
        weighting="roc", context=None, fast=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log(f"Checking data in {data_dir} (downloads NASA archive if missing)")
    download(data_dir, mirror=mirror, subset=subset)
    train, test, rul_true = load_cmapss(subset, data_dir)
    train = add_rul(train, clip=CLIP)
    sensors = informative_sensors(train)
    log(f"Building rolling features for {train.unit.nunique()} training engines")
    F = build_features(train, sensors)
    cols = feature_columns(F)

    # 1. RUL model: grouped cross-validation, then fit on all training engines.
    folds = 3 if fast else 5
    if fast:
        model_mod.MAX_ITER = 100
    log(f"Training RUL models: {folds}-fold cross-validation ({3 * folds} fits, the slowest step)")
    oof = out_of_fold(F, cols, n_splits=folds)
    log("Fitting final RUL models on all training engines")
    model = RULModel().fit(F[cols], F["rul"])
    T_all = build_features(test, sensors)
    T_all = pd.concat([T_all, model.predict(T_all[cols])], axis=1)
    snap = last_cycle(T_all)
    y_true = rul_true.loc[snap.index]
    metrics = {
        "cv_rmse_clipped": rmse(oof["rul"], oof["point"]),
        "test_rmse_clipped": rmse(y_true.clip(upper=CLIP), snap["point"]),
        "test_rmse_unclipped": rmse(y_true, snap["point"]),
        "test_phm_score": phm_score(y_true.clip(upper=CLIP), snap["point"]),
        "test_interval_coverage_80": float(((y_true.clip(upper=CLIP) >= snap["lower"]) &
                                            (y_true.clip(upper=CLIP) <= snap["upper"])).mean()),
    }
    log("Permutation importance")
    imp = model.importance(F[cols].sample(4000, random_state=0), F["rul"].sample(4000, random_state=0))

    # 2. Simulated maintenance log for run-to-failure engines, replayed from out-of-fold predictions.
    log("Simulating maintenance event log and mining it")
    log_hist = eventlog.simulate(oof)
    d = processmining.dfg(log_hist)
    conf = processmining.conformance(log_hist)
    var = processmining.variants(log_hist)
    final = log_hist.groupby("case_id")["activity"].last()
    alert_c = log_hist[log_hist.activity == "Degradation alert"].groupby("case_id")["cycle"].first()
    fail_c = oof.groupby("unit")["cycle"].max().add(1).rename(lambda u: f"engine_{u:03d}")
    metrics.update({
        "policy_planned_removal_share": float((final == "Engine removal (planned)").mean()),
        "policy_unplanned_failure_share": float((final == "Engine failure (unplanned removal)").mean()),
        "policy_mean_alert_to_failure_cycles": float((fail_c.reindex(alert_c.index) - alert_c).mean()),
        "conformance_mean_fitness": float(conf["fitness"].mean()),
    })

    # 3. In-service fleet (test engines): live event log up to the current cycle.
    log_live = eventlog.simulate(T_all[["unit", "cycle", "lower"]], in_service=True)
    last_act = log_live.groupby("case_id")["activity"].last()
    dismissed = log_live[log_live.activity == "Alert dismissed"].groupby("case_id").size()
    pending = (last_act == "Degradation alert").astype(int)
    devs = dismissed.reindex(last_act.index, fill_value=0) + pending

    # 4. Criteria and MCDA ranking. Optional context CSV: column `unit` plus extra criteria
    #    (higher = more urgent), e.g. operational criticality or spare-engine availability.
    log("Ranking the in-service fleet (TOPSIS + weight sensitivity)")
    hi = health_index(test, train, sensors)
    crit = pd.DataFrame({
        "rul_lower": snap["lower"],
        "degradation_rate": degradation_rate(test, hi).reindex(snap.index),
        "process_deviations": devs.rename(lambda c: int(c.split("_")[1])).reindex(snap.index, fill_value=0),
    })
    benefit = dict(CRITERIA)
    if context is not None:
        ctx = pd.read_csv(context).set_index("unit")
        crit = crit.join(ctx)
        benefit.update({c: True for c in ctx.columns})
    w = mcda.roc_weights(list(benefit)) if weighting == "roc" else mcda.critic_weights(crit, benefit)
    rank = mcda.topsis(crit, w, benefit)
    rank = (rank.join(mcda.sensitivity(crit, w, benefit, top=top))
            .join(crit).join(snap[["point", "lower", "upper"]].rename(columns={"point": "rul_point"}))
            .join(y_true))
    k = top
    true_urgent = set(y_true[y_true <= 30].index)
    hit = lambda idx: len(set(idx[:k]) & true_urgent) / k
    metrics.update({
        "mcda_weighting": weighting,
        "mcda_weights": w.round(3).to_dict(),
        f"precision_at_{k}_mcda": hit(rank.index),
        f"precision_at_{k}_rul_only": hit(snap["point"].sort_values().index),
        f"precision_at_{k}_critic_weights": hit(mcda.topsis(crit, mcda.critic_weights(crit, benefit), benefit).index),
        "base_rate_true_rul_le_30": len(true_urgent) / len(y_true),
    })

    # 5. Explanations for the top engines.
    healthy = F[F["rul"] >= CLIP][cols]
    top_feats = list(imp.index[:8])
    lines = []
    for u, row in rank.head(k).iterrows():
        lines.append(f"- {explain.rationale(u, row, crit, w, len(rank))} "
                     f"Sensor evidence: {'; '.join(explain.sensor_deviations(snap.loc[u], healthy, top_feats))}.")
    table = rank.head(k).reset_index()[["unit", "rank", "closeness", "rul_point", "lower", "upper",
                                        *[c for c in crit.columns if c != "rul_lower"], "top_stability"]]
    table = table.rename(columns={"lower": "rul_p10", "upper": "rul_p90"})
    tmd = _md_table(table)
    llm_txt = ""
    if use_llm:
        from . import llm
        llm_txt = llm.recommend(tmd)
        flagged = llm.grounding_check(llm_txt, tmd)
        llm_txt += "\n\n*Grounding check:* " + ("all numbers found in the table." if not flagged
                                                 else f"numbers not in the table: {', '.join(flagged)}")

    # 6. Write outputs.
    log(f"Writing outputs to {out}")
    rank.to_csv(out / "ranking.csv")
    eventlog.to_xes_columns(log_hist).to_csv(out / "eventlog_simulated.csv", index=False)
    conf.to_csv(out / "conformance.csv", index=False)
    var.to_csv(out / "variants.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", header=["importance"])
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    processmining.render_dfg(d, out / "dfg.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    o = np.argsort(y_true.clip(upper=CLIP).values)
    ax.fill_between(range(len(o)), snap["lower"].values[o], snap["upper"].values[o], alpha=0.25, label="80% band")
    ax.plot(snap["point"].values[o], ".", label="predicted RUL")
    ax.plot(y_true.clip(upper=CLIP).values[o], "k-", lw=1, label="true RUL (clipped at 125)")
    ax.set(xlabel="test engines, sorted by true RUL", ylabel="cycles", title=f"C-MAPSS {subset}: RUL on the test fleet")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "rul_test.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    imp.head(12)[::-1].plot.barh(ax=ax)
    ax.set(xlabel="permutation importance (RMSE increase, cycles)", title="Top features")
    fig.tight_layout(); fig.savefig(out / "feature_importance.png", dpi=150); plt.close(fig)

    report = [f"# Maintenance priority report, C-MAPSS {subset}", "",
              "*Sensor data: NASA C-MAPSS (simulated engines). Maintenance event log: SIMULATED by `xpdm.eventlog`.*", "",
              "## Model", "", "```json", json.dumps(metrics, indent=2), "```", "",
              f"## Top {k} engines to inspect", "", tmd, "", "## Why", "", *lines]
    if llm_txt:
        report += ["", "## Generative-AI summary (check against the table)", "", llm_txt]
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    log("Done")
    return metrics, rank


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", default="FD001")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--mirror", default=None, help="base URL holding train_/test_/RUL_ .txt files")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--weights", choices=["roc", "critic"], default="roc",
                    help="roc: rank-order centroid from the stated priority order; critic: data-driven")
    ap.add_argument("--context", default=None, help="CSV with `unit` and extra criteria (higher = more urgent)")
    ap.add_argument("--fast", action="store_true", help="3 folds and 100 boosting rounds: quicker, slightly less accurate")
    ap.add_argument("--llm", action="store_true", help="add a generative-AI summary (needs anthropic + API key)")
    a = ap.parse_args()
    metrics, rank = run(a.subset, a.data_dir, a.out, a.mirror, a.llm, a.top, a.weights, a.context, a.fast)
    print(json.dumps(metrics, indent=2))
    print(rank.head(a.top)[["rank", "closeness", "rul_point", "rul_true", "top_stability"]].to_string())


if __name__ == "__main__":
    main()
