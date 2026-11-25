"""Comparison statistics and figures for the SNAPS/MARS notebook."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO
from scipy import stats

from modules.results import calculate_residue_mismatches


def compare_results(
    results_1_orig: pd.DataFrame, results_2_orig: pd.DataFrame
) -> pd.DataFrame:
    numeric = ['assignments', 'reliable_assignments', 'errors']
    results_1, results_2 = results_1_orig.copy(), results_2_orig.copy()
    name_1 = results_1.attrs['name']
    name_2 = results_2.attrs['name']
    for frame in (results_1, results_2):
        frame['key'] = frame['protein'].str.strip().str.lower()
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors='coerce')
    results_1['condition'] = results_1['condition'].astype(str).str.strip()
    results_2['condition'] = results_2['condition'].astype(str).str.strip()
    # discard every condition in results_2 but the best one
    # results_2 = results_2[results_2["condition"] == "II"].dropna(subset=numeric)
    results_2 = results_2.sort_values(
        'reliable_assignments', ascending=False
    ).drop_duplicates(subset='key', keep='first')
    results_2 = results_2.dropna(subset=numeric)

    def summarise(frame, tool, by):
        aggregations = {
            'protein': ('protein', 'first'),
            'n_assign': ('assignments', 'max'),
            'reliable': ('reliable_assignments', 'mean'),
            'errors': ('errors', 'mean'),
            'n_runs': ('errors', 'size'),
        }
        if 'fasta_file' in frame.columns:
            aggregations['fasta_file'] = ('fasta_file', 'first')

        grouped = frame.groupby(by).agg(**aggregations)
        grouped['correct'] = grouped['n_assign'] - grouped['errors']
        return grouped.add_suffix(f'_{tool}')

    summary_1 = summarise(results_1, name_1, ['key', 'condition']).reset_index(
        level='condition'
    )
    summary_2 = summarise(results_2, name_2, ['key'])
    compared = summary_1.join(summary_2, how='inner').reset_index()
    # Original total length assumption - assume it's the maximum of whichever one is longest
    # compared["N"] = compared[[f"n_assign_{name_1}", f"n_assign_{name_2}"]].max(axis=1)
    # Instead now we'll read the sequence length from results_x['fasta_file'] using biopython
    # And if there is no fasta file then fall back to that

    def sequence_length(path):
        if pd.isna(path):
            return np.nan
        return len(next(SeqIO.parse(path, 'fasta')).seq)

    for tool in (name_1, name_2):
        fasta_col = f'fasta_file_{tool}'
        n_assign_col = f'n_assign_{tool}'
        if fasta_col in compared.columns:
            fasta_lengths = compared[fasta_col].map(sequence_length)
            compared[f'total_{tool}'] = fasta_lengths.fillna(
                compared[n_assign_col]
            )
        else:
            compared[f'total_{tool}'] = compared[n_assign_col]

    compared['N'] = compared[[f'total_{name_1}', f'total_{name_2}']].max(
        axis=1
    )

    for tool in (name_1, name_2):
        compared[f'correct_frac_{tool}'] = (
            compared[f'correct_{tool}'] / compared['N']
        )
        compared[f'reliable_frac_{tool}'] = (
            compared[f'reliable_{tool}'] / compared['N']
        )
        compared[f'err_rate_{tool}'] = (
            compared[f'errors_{tool}'] / compared[f'n_assign_{tool}']
        )
    compared['diff'] = (
        compared[f'correct_frac_{name_1}'] - compared[f'correct_frac_{name_2}']
    )

    def fisher(row):
        total = int(row['N'])
        correct_1 = round(row[f'correct_{name_1}'])
        correct_2 = round(row[f'correct_{name_2}'])

        return stats.fisher_exact(
            [
                [correct_1, total - correct_1],
                [correct_2, total - correct_2],
            ]
        )[1]

    compared['p_fisher'] = compared.apply(fisher, axis=1)
    compared.attrs['name_1'] = name_1
    compared.attrs['name_2'] = name_2
    return compared


def plot_comparisons(compared: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt

    name_1 = compared.attrs['name_1']
    name_2 = compared.attrs['name_2']

    # Keep a stable color assignment by comparison order.
    colors = {name_1: '#1f77b4', name_2: '#d95f02'}
    limit = np.abs(compared['diff']).max() * 1.1
    rng = np.random.default_rng(0)
    plots = {}

    for condition in sorted(compared['condition'].unique()):
        plot = (
            compared[compared['condition'] == condition]
            .sort_values('diff')
            .reset_index(drop=True)
        )
        y = np.arange(len(plot))
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, max(5, 0.45 * len(plot) + 2)),
            gridspec_kw={'width_ratios': [1.4, 1, 1.1]},
        )
        fig.suptitle(
            f'{name_1} condition: {condition} vs {name_2}', fontsize=14, y=1.02
        )

        correct_frac_1 = f'correct_frac_{name_1}'
        correct_frac_2 = f'correct_frac_{name_2}'
        reliable_frac_1 = f'reliable_frac_{name_1}'
        reliable_frac_2 = f'reliable_frac_{name_2}'
        err_rate_1 = f'err_rate_{name_1}'
        err_rate_2 = f'err_rate_{name_2}'
        protein_col = f'protein_{name_1}'

        ax = axes[0]
        ax.hlines(
            y,
            plot[correct_frac_2],
            plot[correct_frac_1],
            color='grey',
            alpha=0.5,
        )
        ax.scatter(
            plot[correct_frac_1],
            y,
            color=colors[name_1],
            s=60,
            label=name_1,
        )
        ax.scatter(
            plot[correct_frac_2],
            y,
            color=colors[name_2],
            s=60,
            label=name_2,
        )
        for yi, p_value in zip(y, plot['p_fisher']):
            star = (
                '***'
                if p_value < 0.001
                else '**'
                if p_value < 0.01
                else '*'
                if p_value < 0.05
                else ''
            )
            if star:
                ax.text(1.02, yi, star, va='center', ha='left')
        ax.set(
            yticks=y,
            yticklabels=plot[protein_col],
            xlim=(0, 1.08),
            xlabel='Correct assignments / total residues',
            title='A. Correct fraction per protein\n(* = Fisher p<0.05, approximate)',
        )
        ax.legend(loc='lower right')
        ax.grid(axis='x', alpha=0.3)

        ax = axes[1]
        ax.barh(
            y,
            plot['diff'],
            color=[
                colors[name_1] if value > 0 else colors[name_2]
                for value in plot['diff']
            ],
            alpha=0.85,
        )
        differences = plot['diff'].to_numpy()
        bootstrap = rng.choice(
            differences, size=(10000, len(differences)), replace=True
        ).mean(axis=1)
        lo, hi = np.percentile(bootstrap, [2.5, 97.5])
        ax.axvspan(
            lo,
            hi,
            color='grey',
            alpha=0.2,
            label=f'Bootstrap 95% CI\n({lo:.2f} to {hi:.2f})',
        )
        ax.axvline(0, color='k', lw=0.8)
        ax.axvline(
            differences.mean(),
            color='k',
            ls='--',
            label=f'Mean diff = {differences.mean():.2f}',
        )
        ax.set(
            yticks=y,
            yticklabels=[],
            xlim=(-limit, limit),
            xlabel=f'{name_1} - {name_2} (correct fraction)',
            title=f'B. Difference (>0: {name_1} better)',
        )
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        ax = axes[2]
        for tool, correct_frac, err_col, offset in (
            (name_1, correct_frac_1, err_rate_1, (5, 5)),
            (name_2, correct_frac_1, err_rate_2, (5, -8)),
        ):
            ax.scatter(
                plot[correct_frac],
                plot[err_col],
                color=colors[tool],
                s=60,
                alpha=0.85,
                label=tool,
            )
            for _, row in plot.iterrows():
                ax.annotate(
                    row[protein_col],
                    (row[correct_frac], row[err_col]),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=7,
                    color=colors[tool],
                )

        for _, row in plot.iterrows():
            ax.plot(
                [row[correct_frac_1], row[correct_frac_2]],
                [row[err_rate_1], row[err_rate_2]],
                color='grey',
                alpha=0.3,
            )

        ax.set(
            xlabel='Correct assignments',
            ylabel='Errors',
            title='C. Correctness vs error rate\n(lower right is best)',
            ylim=(0.0, 1.0),
            xlim=(0.0, 1.0),
        )
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plots[condition] = fig

    return plots


def plot_comparisons_big(compared: pd.DataFrame) -> dict:
    import matplotlib.pyplot as plt

    name_1 = compared.attrs['name_1']
    name_2 = compared.attrs['name_2']
    colors = {name_1: '#1f77b4', name_2: '#d95f02'}
    plots = {}

    correct_frac_1 = f'correct_frac_{name_1}'
    correct_frac_2 = f'correct_frac_{name_2}'
    err_rate_1 = f'err_rate_{name_1}'
    err_rate_2 = f'err_rate_{name_2}'

    for condition in sorted(compared['condition'].unique()):
        plot = (
            compared[compared['condition'] == condition]
            .sort_values('diff')
            .reset_index(drop=True)
        )
        differences = plot['diff'].dropna().to_numpy()
        if len(differences) == 0:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f'{name_1} condition: {condition} vs {name_2}', fontsize=14
        )

        # A. Paired correct fractions, with proteins overlaid in a compact panel.
        ax = axes[0]
        y = np.linspace(0, 1, len(plot)) if len(plot) > 1 else np.array([0.5])
        for yi, (_, row) in zip(y, plot.iterrows()):
            ax.plot(
                [row[correct_frac_2], row[correct_frac_1]],
                [yi, yi],
                color='grey',
                alpha=0.18,
                lw=0.7,
                zorder=1,
            )

        ax.scatter(
            plot[correct_frac_1],
            y,
            color=colors[name_1],
            s=14,
            alpha=0.65,
            label=name_1,
            zorder=2,
        )
        ax.scatter(
            plot[correct_frac_2],
            y,
            color=colors[name_2],
            s=14,
            alpha=0.65,
            label=name_2,
            zorder=2,
        )

        for yi, p_value in zip(y, plot['p_fisher']):
            if pd.isna(p_value):
                continue
            star = (
                '***'
                if p_value < 0.001
                else '**'
                if p_value < 0.01
                else '*'
                if p_value < 0.05
                else ''
            )
            if star:
                ax.text(
                    1.02,
                    yi,
                    star,
                    va='center',
                    ha='left',
                    fontsize=7,
                    clip_on=False,
                )

        ax.set(
            xlim=(0, 1.08),
            ylim=(-0.03, 1.03),
            xlabel='Correct assignments / total residues',
            ylabel='Proteins (ranked by difference)',
            title='A. Correct fraction per protein\n(* = Fisher p<0.05, approximate)',
        )
        ax.set_yticks([])
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        # B. Distribution of per-protein differences and bootstrap CI for the mean.
        ax = axes[1]
        rng = np.random.default_rng(0)
        bootstrap = rng.choice(
            differences, size=(10_000, len(differences)), replace=True
        ).mean(axis=1)
        lo, hi = np.percentile(bootstrap, [2.5, 97.5])
        limit = max(np.abs(differences).max() * 1.1, 0.05)

        ax.hist(differences, bins='auto', color='grey', alpha=0.65)
        ax.axvspan(
            lo,
            hi,
            color='gold',
            alpha=0.25,
            label=f'Bootstrap 95% CI\n({lo:.2f} to {hi:.2f})',
        )
        ax.axvline(0, color='k', lw=0.8)
        ax.axvline(
            differences.mean(),
            color='k',
            ls='--',
            label=f'Mean diff = {differences.mean():.2f}',
        )
        ax.set(
            xlim=(-limit, limit),
            xlabel=f'{name_1} - {name_2} (correct fraction)',
            ylabel='Number of proteins',
            title=f'B. Distribution of differences\n(>0: {name_1} better)',
        )
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='x', alpha=0.3)

        # C. Correctness versus error rate, with paired points connected.
        ax = axes[2]
        ax.scatter(
            plot[correct_frac_1],
            plot[err_rate_1],
            color=colors[name_1],
            s=18,
            alpha=0.65,
            label=name_1,
        )
        ax.scatter(
            plot[correct_frac_2],
            plot[err_rate_2],
            color=colors[name_2],
            s=18,
            alpha=0.65,
            label=name_2,
        )
        for _, row in plot.iterrows():
            ax.plot(
                [row[correct_frac_1], row[correct_frac_2]],
                [row[err_rate_1], row[err_rate_2]],
                color='grey',
                alpha=0.2,
                lw=0.7,
            )

        ax.set(
            xlabel='Correct assignments',
            ylabel='Errors',
            title='C. Correctness vs error rate\n(lower right is best)',
            xlim=(0, 1),
            ylim=(0, 1),
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        plots[condition] = fig

    return plots


def analyse_mismatches(mismatches):
    amino_acids = [
        'ALA',
        'ARG',
        'ASN',
        'ASP',
        'CYS',
        'GLN',
        'GLU',
        'GLY',
        'HIS',
        'ILE',
        'LEU',
        'LYS',
        'MET',
        'PHE',
        'PRO',
        'SER',
        'THR',
        'TRP',
        'TYR',
        'VAL',
    ]
    amino_acid_set = set(amino_acids)
    figures = {}

    def plot_counts(ax, counts, title, *, limit=20, sort_by_value=True):
        if not counts:
            ax.set_title(title)
            ax.text(
                0.5,
                0.5,
                'No data',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
            return

        items = list(counts.items())
        if sort_by_value:
            items.sort(key=lambda item: item[1], reverse=True)
        else:
            items.sort(key=lambda item: item[0])

        items = items[:limit]
        labels, values = zip(*items)
        labels = [str(label) for label in labels]

        ax.bar(labels, values, color='#4C78A8')
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', labelrotation=45)
        ax.grid(axis='y', alpha=0.25)
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
            sequence_length = df.attrs['protein_length']

            err_residue.update(
                name
                for name in df['residue_name'].dropna()
                if name in amino_acid_set
            )
            real_residue.update(
                name
                for name in df['reference_residue_name'].dropna()
                if name in amino_acid_set
            )

            for residue, reference in zip(
                df['residue_name'], df['reference_residue_name']
            ):
                if residue in amino_acid_set and reference in amino_acid_set:
                    real_err_res_pair[(residue, reference)] += 1

            res = df['residue_number'].dropna()
            ref = df['reference_number'].dropna()

            res_position_pct.extend(res / sequence_length * 100)
            reference_position_pct.extend(ref / sequence_length * 100)

            numbers = df[['residue_number', 'reference_number']].dropna()
            distance_gap_pct.extend(
                abs(numbers['residue_number'] - numbers['reference_number'])
                / sequence_length
                * 100
            )

        fig, axes = plt.subplots(2, 3, figsize=(17, 10))
        fig.suptitle(f'Mismatch analysis: {condition_name}', fontsize=16)

        plot_counts(axes[0, 0], err_residue, 'Mismatch residue')
        plot_counts(axes[0, 1], real_residue, 'Reference residue')

        pair_matrix = np.zeros((len(amino_acids), len(amino_acids)))
        aa_index = {aa: i for i, aa in enumerate(amino_acids)}
        for (residue, reference), count in real_err_res_pair.items():
            pair_matrix[aa_index[residue], aa_index[reference]] = count

        image = axes[0, 2].imshow(pair_matrix, cmap='Blues', aspect='auto')
        axes[0, 2].set_title('Mismatch vs. reference residue')
        axes[0, 2].set_xlabel('Reference residue')
        axes[0, 2].set_ylabel('Mismatch residue')
        axes[0, 2].set_xticks(
            range(len(amino_acids)), amino_acids, rotation=90, fontsize=7
        )
        axes[0, 2].set_yticks(range(len(amino_acids)), amino_acids, fontsize=7)
        fig.colorbar(image, ax=axes[0, 2], label='Count')

        axes[1, 0].hist(
            res_position_pct,
            bins=np.arange(0, 110, 10),
            color='#4C78A8',
            edgecolor='white',
        )
        axes[1, 0].set(
            title='Mismatch position in protein',
            xlabel='Position (%)',
            ylabel='Count',
        )

        axes[1, 1].hist(
            reference_position_pct,
            bins=np.arange(0, 110, 10),
            color='#4C78A8',
            edgecolor='white',
        )
        axes[1, 1].set(
            title='Reference position in protein',
            xlabel='Position (%)',
            ylabel='Count',
        )

        gap_bins = np.arange(0, 110, 10)
        axes[1, 2].hist(
            distance_gap_pct, bins=gap_bins, color='#4C78A8', edgecolor='white'
        )
        axes[1, 2].set(
            title='Residue − reference distance (% of protein length)',
            xlabel='Signed distance (%)',
            ylabel='Count',
        )

        fig.tight_layout(rect=(0, 0, 1, 0.96))
        figures[condition_name] = fig

    return figures


def analyse_rci(all_files: dict, BMRB_NAMES: Optional[dict] = None):
    by_bmrb = _analyse_rci(all_files)

    conditions = sorted(
        {
            condition
            for protein_conditions in by_bmrb.values()
            for condition in protein_conditions
        }
    )

    rci_analysis = {
        condition: {
            bmrb_id: protein_conditions[condition]
            for bmrb_id, protein_conditions in by_bmrb.items()
            if condition in protein_conditions
        }
        for condition in conditions
    }

    secondary_structures = ('H', 'E', 'C')
    colors = {'H': 'tab:blue', 'E': 'tab:orange', 'C': 'tab:green'}
    figures = {}

    for condition, proteins in rci_analysis.items():
        bmrb_ids = list(proteins)
        if BMRB_NAMES is not None:
            bmrb_names = list([BMRB_NAMES[x] for x in bmrb_ids])
        else:
            bmrb_names = bmrb_ids
        x = np.arange(len(bmrb_ids))
        width = 0.25

        fig, (ax_individual, ax_collated) = plt.subplots(
            1, 2, figsize=(14, 5), constrained_layout=True
        )
        fig.suptitle(f'RCI mismatch proportions: {condition}')

        # Per-protein grouped bars
        for i, ss in enumerate(secondary_structures):
            values = [proteins[bmrb_id].get(ss, 0.0) for bmrb_id in bmrb_ids]
            ax_individual.bar(
                x + (i - 1) * width,
                values,
                width,
                label=ss,
                color=colors[ss],
            )

        ax_individual.set_title('By protein')
        ax_individual.set_xlabel('Protein name')
        ax_individual.set_ylabel('Mismatch proportion')
        ax_individual.set_xticks(x)
        ax_individual.set_xticklabels(bmrb_names, rotation=45, ha='right')
        ax_individual.set_ylim(0, 1)
        ax_individual.legend(title='Secondary structure')

        # Across-protein mean with standard deviation
        means = []
        standard_deviations = []
        for ss in secondary_structures:
            values = np.array(
                [proteins[bmrb_id].get(ss, 0.0) for bmrb_id in bmrb_ids],
                dtype=float,
            )
            means.append(values.mean() if values.size else 0.0)
            standard_deviations.append(
                values.std(ddof=1) if values.size > 1 else 0.0
            )

        ss_x = np.arange(len(secondary_structures))
        ax_collated.bar(
            ss_x,
            means,
            yerr=standard_deviations,
            capsize=5,
            color=[colors[ss] for ss in secondary_structures],
        )
        ax_collated.set_title('Across proteins')
        ax_collated.set_xlabel('Secondary structure')
        ax_collated.set_ylabel('Mean mismatch proportion ± SD')
        ax_collated.set_xticks(ss_x)
        ax_collated.set_xticklabels(secondary_structures)
        ax_collated.set_ylim(0, 1)

        figures[condition] = fig

    return figures


def _analyse_rci(all_files: dict) -> dict:
    """

    RCI - random coil index. Psipred kinda does this?
    (base) michiko@michiko-HP-Z4-G4-Workstation:~/repos/HP-SNAPS/notebooks/mars_test_set/4136$ cat *psipred*
    # PSIPRED HFORMAT (PSIPRED V4.0)

    Conf: 998201107899999966787843130654545789999999999999999424243999
    Pred: CCCCCHHCHHHHHHHHHHHHHHHCCCHHHHCCCHHHHHHHHHHHHHHHHHHHHCCHHHHH
      AA: MNPYIYLGGAILAEVIGTTLMKFSEGFTRLWPSVGTIICYCASFWLLAQTLAYIPTGIAY
                  10        20        30        40        50        60


    Conf: 99860789999898314323025878999999999999999998218999
    Pred: HHHHCHHHHHHHHHHCCCCCCCCCHHHHHHHHHHHHHHHHHHHHHCCCCC
      AA: AIWSGVGIVLISLLSWGFFGQRLDLPAIIGMMLICAGVLIINLLSRSTPH
                  70        80        90       100       110

    H = alpha helxi
    E = beta strand
    C = random coil
    Conf = confidence

    We need to align this to the errors, see whether most of the errors are in coils, helices, or sheets
    """
    output = {}

    for bmrb_id, protein in all_files.items():
        output[bmrb_id] = {}

        for condition, files in protein.items():
            psipred = read_psipred(files['psipred_file'])
            protein_length = len(psipred)
            mismatches = calculate_residue_mismatches(
                files['result_file'],
                files['nef_file'],
                flank=0,
            )

            # Convert the mismatch reference positions to a set for fast lookup.
            mismatch_positions = mismatches['reference_number']

            counts = {'H': 0, 'E': 0, 'C': 0}

            for residue in psipred:
                ss = residue['secondary_structure']
                if residue['reference_number'] in mismatch_positions:
                    counts[ss] += 1

            proportions = {}
            for x, y in counts.items():
                if y > 0 and sum(counts.values()) > 0:
                    proportions[x] = y / sum(counts.values())
                else:
                    proportions[x] = 0.0

            if sum([y for y in proportions.values()]) > 0:
                output[bmrb_id][condition] = proportions

    return output


def read_psipred(file_path: str) -> list[dict]:
    """Read a PSIPRED HFORMAT file into one record per sequence position."""
    records = []
    confidence = prediction = sequence = None

    def flush_block():
        nonlocal confidence, prediction, sequence

        if confidence is None and prediction is None and sequence is None:
            return

        if confidence is None or prediction is None or sequence is None:
            raise ValueError(f'Incomplete PSIPRED block in {file_path}')

        if not (len(confidence) == len(prediction) == len(sequence)):
            raise ValueError(
                f'PSIPRED block lengths differ in {file_path}: '
                f'Conf={len(confidence)}, Pred={len(prediction)}, AA={len(sequence)}'
            )

        start = len(records) + 1
        for offset, (conf, ss, aa) in enumerate(
            zip(confidence, prediction, sequence)
        ):
            records.append(
                {
                    'reference_number': start + offset,
                    'residue_name': aa,
                    'secondary_structure': ss,
                    'confidence': int(conf),
                }
            )

        confidence = prediction = sequence = None

    with open(file_path, encoding='utf-8') as handle:
        for line in handle:
            match = re.match(r'^\s*(Conf|Pred|AA):\s*(\S+)', line)
            if not match:
                continue

            label, value = match.groups()
            if label == 'Conf':
                # A new Conf line starts the next block.
                flush_block()
                confidence = value
            elif label == 'Pred':
                prediction = value
            else:
                sequence = value

        flush_block()

    return records
