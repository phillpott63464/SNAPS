"""Comparison statistics and figures for the SNAPS/MARS notebook."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
import matplotlib.pyplot as plt


def compare_results(snaps_results: pd.DataFrame, mars_results: pd.DataFrame) -> pd.DataFrame:
    numeric = ["assignments", "reliable_assignments", "errors"]
    snaps, mars = snaps_results.copy(), mars_results.copy()
    for frame in (snaps, mars):
        frame["key"] = frame["protein"].str.strip().str.lower()
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    snaps["condition"] = snaps["condition"].astype(str).str.strip()
    mars["condition"] = mars["condition"].astype(str).str.strip()
    # discard every condition but condition II
    mars = mars[mars["condition"] == "II"].dropna(subset=numeric)
    mars = (
        mars.sort_values("reliable_assignments", ascending=False)
        .drop_duplicates(subset="key", keep="first")
    )
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

def analyse_mismatches(mismatches):
    amino_acids = [
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
    ]
    amino_acid_set = set(amino_acids)
    figures = []

    def plot_counts(ax, counts, title, *, limit=20, sort_by_value=True):
        if not counts:
            ax.set_title(title)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            return

        items = list(counts.items())
        if sort_by_value:
            items.sort(key=lambda item: item[1], reverse=True)
        else:
            items.sort(key=lambda item: item[0])

        items = items[:limit]
        labels, values = zip(*items)
        labels = [str(label) for label in labels]

        ax.bar(labels, values, color="#4C78A8")
        ax.set_title(title)
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", labelrotation=45)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

    for condition_name, proteins in mismatches.items():
        err_residue = Counter()
        real_residue = Counter()
        real_err_res_pair = Counter()
        res_number = Counter()
        reference_number = Counter()
        distance_gap = Counter()

        res_position_pct = []
        reference_position_pct = []
        distance_gap_pct = []

        for df in proteins.values():
            print(df)
            sequence_length = df.attrs["protein_length"]

            err_residue.update(
                name for name in df["residue_name"].dropna()
                if name in amino_acid_set
            )
            real_residue.update(
                name for name in df["reference_residue_name"].dropna()
                if name in amino_acid_set
            )

            for residue, reference in zip(
                df["residue_name"], df["reference_residue_name"]
            ):
                if residue in amino_acid_set and reference in amino_acid_set:
                    real_err_res_pair[(residue, reference)] += 1

            res = df["residue_number"].dropna()
            ref = df["reference_number"].dropna()

            res_position_pct.extend(res / sequence_length * 100)
            reference_position_pct.extend(ref / sequence_length * 100)

            numbers = df[["residue_number", "reference_number"]].dropna()
            distance_gap_pct.extend(
                abs(numbers["residue_number"] - numbers["reference_number"])
                / sequence_length * 100
            )

        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        fig.suptitle(f"Mismatch analysis: {condition_name}", fontsize=16)

        plot_counts(axes[0, 0], err_residue, "Mismatch residue")
        plot_counts(axes[0, 1], real_residue, "Reference residue")

        pair_matrix = np.zeros((len(amino_acids), len(amino_acids)))
        aa_index = {aa: i for i, aa in enumerate(amino_acids)}
        for (residue, reference), count in real_err_res_pair.items():
            pair_matrix[aa_index[residue], aa_index[reference]] = count

        image = axes[0, 2].imshow(pair_matrix, cmap="Blues", aspect="auto")
        axes[0, 2].set_title("Mismatch vs. reference residue")
        axes[0, 2].set_xlabel("Reference residue")
        axes[0, 2].set_ylabel("Mismatch residue")
        axes[0, 2].set_xticks(range(len(amino_acids)), amino_acids,
                               rotation=90, fontsize=7)
        axes[0, 2].set_yticks(range(len(amino_acids)), amino_acids,
                               fontsize=7)
        fig.colorbar(image, ax=axes[0, 2], label="Count")

        axes[1, 0].hist(res_position_pct, bins=np.arange(0, 110, 10),
                        color="#4C78A8", edgecolor="white")
        axes[1, 0].set(title="Mismatch position in protein", xlabel="Position (%)",
                       ylabel="Count")

        axes[1, 1].hist(reference_position_pct, bins=np.arange(0, 110, 10),
                        color="#4C78A8", edgecolor="white")
        axes[1, 1].set(title="Reference position in protein", xlabel="Position (%)",
                       ylabel="Count")

        gap_bins = np.arange(0, 110, 10)
        axes[1, 2].hist(distance_gap_pct, bins=gap_bins,
                        color="#4C78A8", edgecolor="white")
        axes[1, 2].set(title="Residue − reference distance (% of protein length)",
                       xlabel="Signed distance (%)", ylabel="Count")

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figures.append(fig)

    return figures
