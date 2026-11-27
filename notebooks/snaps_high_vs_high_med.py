import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    from modules.analysis import compare_results, plot_comparisons
    from modules.bmrb import run_test_set
    from modules.results import assignment_table

    # from modules.pdb import colour_mars_set
    return (
        Path,
        assignment_table,
        compare_results,
        plot_comparisons,
        run_test_set,
    )


@app.cell
def _(Path):
    BMRB_IDS = [
        # 5471,
        # 4354,
        # 4384,
        # 4022,
        # 4106,
        4457,
        4402,
        4341,
        547,
        4082,
        4136,
        5387,
    ]
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
    DATA_DIR = Path(__file__).resolve().parent / 'mars_test_set'
    OUTPUT_DIR = Path(__file__).resolve().parent / 'mars_test_result'
    return BMRB_IDS, BMRB_NAMES, DATA_DIR, OUTPUT_DIR


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    plot_comparisons,
    run_test_set,
):
    snaps_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    snaps_results = assignment_table(snaps_test_files, BMRB_NAMES, 'snaps')
    snaps_results = snaps_results[snaps_results['condition'] == 'triplet']
    snaps_results_high_med = assignment_table(
        snaps_test_files,
        BMRB_NAMES,
        'snaps_high_med',
        good_confidences=['High', 'Medium'],
    )
    comparison_high_med = compare_results(
        snaps_results_high_med, snaps_results
    )
    plots_high_med = plot_comparisons(comparison_high_med)
    return (plots_high_med,)


@app.cell(hide_code=True)
def _(mo, plots_high_med):
    mo.md(
        rf"""
    ## SNAPS High vs High and Medium

    SNAPS gives confidence scores to its assigned residues. High, Medium, Unreliable, Undefined. 

    Hypothesis: High and Medium should assign more residues than High alone
    Conclusion: Seems like it yeah

    ### 5.1 Direct Strategy
    {mo.as_html(plots_high_med['direct'])}

    ### 5.2 Iterative Strategy
    {mo.as_html(plots_high_med['iterative'])}

    ### 5.3 Triplet Strategy
    {mo.as_html(plots_high_med['triplet'])}
    """
    )


@app.cell
def _(mo, plots_high_med):
    mo.vstack([mo.as_html(plot) for plot in plots_high_med.values()])


if __name__ == '__main__':
    app.run()
