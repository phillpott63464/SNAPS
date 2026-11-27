"""Reproduce the MARS comparison on the public SNAPS test set.

The notebook deliberately keeps orchestration and presentation here. Parsers,
downloaders, statistics, and plotting live in ``notebooks.modules`` so they can
be reused by other analyses.
"""

import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    from modules.analysis import compare_results, plot_comparisons_big
    from modules.mars import get_mars_data
    from modules.rbmrb import get_rbmrb
    from modules.results import assignment_table
    from modules.test_sets import (
        run_triplet_default_analyzed,
    )

    return (
        Path,
        assignment_table,
        compare_results,
        get_mars_data,
        get_rbmrb,
        mo,
        plot_comparisons_big,
        run_triplet_default_analyzed,
    )


@app.cell
def _(Path, get_rbmrb):
    DATA_DIR = Path(__file__).resolve().parent / 'mars_test_set'
    OUTPUT_DIR = Path(__file__).resolve().parent / 'mars_test_result'
    BMRB_IDS = [
        5471,
        4354,
        4384,
        4022,
        4106,
        4457,
        4402,
        4341,
        547,
        4082,
        4136,
        5387,
        19714,
    ] + get_rbmrb(data_dir=DATA_DIR, number=200)
    BMRB_IDS = BMRB_IDS
    BMRB_NAMES = {
        5471: 'Malate synthase G',
        4354: 'Maltose binding protein',
        4384: 'Rous Sarcoma Virus capsid',
        4022: 'Human carbonic anhydrase I',
        4106: 'N-terminal domain of enzyme I (EIN)',
        4457: 'E-cadherin domains II and III',
        4402: 'Human prion protein',
        4341: 'Superoxide dismutase',
        547: 'Calmodulin/M13',
        4082: 'Profilin',
        4136: 'E. coli EmrE',
        5387: 'Human ubiquitin',
    }
    for key in BMRB_IDS:
        BMRB_NAMES.setdefault(key, str(key))
    return BMRB_IDS, BMRB_NAMES, DATA_DIR, OUTPUT_DIR


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    run_triplet_default_analyzed,
):
    # Published MARS counts used in the original benchmark.

    snaps_test_files = run_triplet_default_analyzed(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR / 'default'
    )
    snaps_results_high = assignment_table(
        snaps_test_files, BMRB_NAMES, 'snaps', good_confidences=['High']
    )
    snaps_results_high_med = assignment_table(
        snaps_test_files,
        BMRB_NAMES,
        'snaps',
        good_confidences=['High', 'Medium'],
    )
    return snaps_results_high, snaps_results_high_med


@app.cell
def _(snaps_results_high):
    print(len(snaps_results_high['bmrb_id'].unique()))


@app.cell
def _(BMRB_IDS, BMRB_NAMES, DATA_DIR, get_mars_data):
    mars_results = get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR)
    return (mars_results,)


@app.cell
def _(mars_results):
    print(len(mars_results['bmrb_id'].unique()))


@app.cell
def _(
    compare_results,
    mars_results,
    plot_comparisons_big,
    snaps_results_high,
    snaps_results_high_med,
):
    comparison_high = compare_results(snaps_results_high, mars_results)
    comparison_high_med = compare_results(snaps_results_high_med, mars_results)
    plots_high = plot_comparisons_big(comparison_high)
    plots_high_med = plot_comparisons_big(comparison_high_med)
    return comparison_high, plots_high, plots_high_med


@app.cell
def _(comparison_high):
    print(len(comparison_high['protein_snaps'].unique()))


@app.cell
def _(mo, plots_high):
    mo.vstack([mo.as_html(plot) for plot in plots_high.values()])


@app.cell
def _(mo, plots_high_med):
    mo.vstack([mo.as_html(plot) for plot in plots_high_med.values()])


@app.cell
def _(comparison_high):
    comparison_high.to_clipboard()


if __name__ == '__main__':
    app.run()
