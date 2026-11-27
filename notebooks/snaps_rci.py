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

    from modules.analysis import analyse_rci
    from modules.bmrb import run_test_set
    from modules.rbmrb import get_rbmrb

    return Path, analyse_rci, get_rbmrb, run_test_set


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
    ] + get_rbmrb(DATA_DIR, 50)
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
def _(BMRB_IDS, DATA_DIR, OUTPUT_DIR, run_test_set):
    snaps_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    return (snaps_test_files,)


@app.cell
def _(BMRB_NAMES, analyse_rci, snaps_test_files):
    rci_analysis = analyse_rci(snaps_test_files, BMRB_NAMES)
    return (rci_analysis,)


@app.cell(hide_code=True)
def _(mo, rci_analysis):
    mo.md(
        rf"""
    # Test SNAPS against RCI 

    Testing SNAPS against secondary structure using psipred

    Hypothesis: SNAPS should probably struggle with random coils more because random coils have very similar chemical shifts.

    Conclusion: Needs more data

    ## Direct

    {mo.as_html(rci_analysis['direct'])}

    ## Iterative

    {mo.as_html(rci_analysis['iterative'])}

    ## Triplet

    {mo.as_html(rci_analysis['triplet'])}
    """
    )


if __name__ == '__main__':
    app.run()
