"""Comparison statistics and figures for the SNAPS/MARS notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compare_results(snaps_results: pd.DataFrame, mars_results: pd.DataFrame) -> pd.DataFrame:
    numeric = ["assignments", "reliable_assignments", "errors"]
    snaps, mars = snaps_results.copy(), mars_results.copy()
    for frame in (snaps, mars):
        frame["key"] = frame["protein"].str.strip().str.lower()
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    snaps["condition"] = snaps["condition"].astype(str).str.strip()
    mars = mars.dropna(subset=numeric)

    def summarise(frame, tool, by):
        grouped = frame.groupby(by).agg(protein=("protein", "first"), n_assign=("assignments", "max"),
                                        reliable=("reliable_assignments", "mean"), errors=("errors", "mean"),
                                        n_runs=("errors", "size"))
        grouped["correct"] = grouped["reliable"] - grouped["errors"]
        return grouped.add_suffix(f"_{tool}")

    snap_summary = summarise(snaps, "snaps", ["key", "condition"]).reset_index(level="condition")
    mars_summary = summarise(mars, "mars", ["key"])
    compared = snap_summary.join(mars_summary, how="inner").reset_index()
    compared["N"] = compared[["n_assign_snaps", "n_assign_mars"]].max(axis=1)
    for tool in ("snaps", "mars"):
        compared[f"correct_frac_{tool}"] = compared[f"correct_{tool}"] / compared["N"]
        compared[f"reliable_frac_{tool}"] = compared[f"reliable_{tool}"] / compared["N"]
        compared[f"err_rate_{tool}"] = compared[f"errors_{tool}"] / compared[f"reliable_{tool}"]
    compared["diff"] = compared["correct_frac_snaps"] - compared["correct_frac_mars"]

    def fisher(row):
        total = int(row.N)
        return stats.fisher_exact([[round(row.correct_snaps), total - round(row.correct_snaps)],
                                   [round(row.correct_mars), total - round(row.correct_mars)]])[1]
    compared["p_fisher"] = compared.apply(fisher, axis=1)
    return compared


def plot_comparisons(compared: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    colors = {"snaps": "#1f77b4", "mars": "#d95f02"}
    limit = np.abs(compared["diff"]).max() * 1.1
    rng = np.random.default_rng(0)
    plots = []
    for condition in sorted(compared["condition"].unique()):
        plot = compared[compared["condition"] == condition].sort_values("diff").reset_index(drop=True)
        y = np.arange(len(plot))
        fig, axes = plt.subplots(1, 3, figsize=(18, max(5, 0.45 * len(plot) + 2)),
                                 gridspec_kw={"width_ratios": [1.4, 1, 1.1]})
        fig.suptitle(f"SNAPS condition: {condition} vs MARS", fontsize=14, y=1.02)

        ax = axes[0]
        ax.hlines(y, plot.correct_frac_mars, plot.correct_frac_snaps, color="grey", alpha=.5)
        ax.scatter(plot.correct_frac_snaps, y, color=colors["snaps"], s=60, label="SNAPS")
        ax.scatter(plot.correct_frac_mars, y, color=colors["mars"], s=60, label="MARS")
        for yi, p_value in zip(y, plot.p_fisher):
            star = "***" if p_value < .001 else "**" if p_value < .01 else "*" if p_value < .05 else ""
            if star:
                ax.text(1.02, yi, star, va="center", ha="left")
        ax.set(yticks=y, yticklabels=plot["protein_snaps"], xlim=(0, 1.08),
               xlabel="Correct assignments / total residues",
               title="A. Correct fraction per protein\n(* = Fisher p<0.05, approximate)")
        ax.legend(loc="lower right"); ax.grid(axis="x", alpha=.3)

        ax = axes[1]
        ax.barh(y, plot["diff"], color=[colors["snaps"] if value > 0 else colors["mars"] for value in plot["diff"]], alpha=.85)
        differences = plot["diff"].to_numpy()
        bootstrap = rng.choice(differences, size=(10000, len(differences)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bootstrap, [2.5, 97.5])
        ax.axvspan(lo, hi, color="grey", alpha=.2, label=f"Bootstrap 95% CI\n({lo:.2f} to {hi:.2f})")
        ax.axvline(0, color="k", lw=.8); ax.axvline(differences.mean(), color="k", ls="--", label=f"Mean diff = {differences.mean():.2f}")
        ax.set(yticks=y, yticklabels=[], xlim=(-limit, limit), xlabel="SNAPS - MARS (correct fraction)", title="B. Difference (>0: SNAPS better)")
        ax.legend(loc="lower right", fontsize=8); ax.grid(axis="x", alpha=.3)

        ax = axes[2]
        for tool in ("snaps", "mars"):
            ax.scatter(plot[f"reliable_frac_{tool}"], plot[f"err_rate_{tool}"], color=colors[tool], s=60, alpha=.85, label=tool.upper())
        for _, row in plot.iterrows():
            ax.plot([row.reliable_frac_snaps, row.reliable_frac_mars], [row.err_rate_snaps, row.err_rate_mars], color="grey", alpha=.3)
        ax.set(xlabel="Reliable assignments / total residues", ylabel="Errors / reliable assignments",
               title="C. Completeness vs error rate\n(lower right is best)")
        ax.legend(); ax.grid(alpha=.3)
        plt.tight_layout()
        plots.append(fig)
    return plots