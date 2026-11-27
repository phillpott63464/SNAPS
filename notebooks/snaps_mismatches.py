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

    from modules.analysis import analyse_mismatches
    from modules.bmrb import run_test_set
    from modules.rbmrb import get_rbmrb
    from modules.results import all_mismatches

    # from modules.pdb import colour_mars_set
    return Path, all_mismatches, analyse_mismatches, get_rbmrb, run_test_set


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
    ] + get_rbmrb(DATA_DIR, 200)
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
    DATA_DIR,
    OUTPUT_DIR,
    all_mismatches,
    analyse_mismatches,
    run_test_set,
):
    snaps_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    mismatches = all_mismatches(snaps_test_files, DATA_DIR, OUTPUT_DIR)
    mismatch_figs = analyse_mismatches(mismatches)
    return mismatch_figs, mismatches


@app.cell(hide_code=True)
def _(mismatch_figs, mo):
    mo.md(
        rf"""
    ## Test Mismatches

    What residues go wrong, and where in the protein do they go wrong?

    Hypotheses:
    1) Mismatches will majority be within a few residues of each other
    2) Mismatches will mostly be between residues of similar types
    3) More mismatches will occur around the start and ends of the protein

    Conclusions: 
    1) Seems to be mostly within 20% of the real residue
    2) Definetly have a diagonal in triplet
    3) Not really? Seems to be random.

    ### Direct Strategy
    {mo.as_html(mismatch_figs['direct'])}

    ### Iterative Strategy
    {mo.as_html(mismatch_figs['iterative'])}

    ### Triplet Strategy
    {mo.as_html(mismatch_figs['triplet'])}
    """
    )


@app.cell
def _(BMRB_NAMES, mismatches, mo):
    def _():
        condition_blocks = []

        for condition, protein_results in mismatches.items():
            no_error_proteins = []
            protein_tables = []

            for bmrb_id, df in protein_results.items():
                protein = BMRB_NAMES[bmrb_id]

                if df.empty:
                    no_error_proteins.append(protein)
                else:
                    protein_tables.extend(
                        [
                            mo.md(f'#### {protein} · {len(df)} rows'),
                            mo.ui.table(df),
                        ]
                    )

            content = []
            if no_error_proteins:
                names = ', '.join(no_error_proteins)
                content.append(mo.md(f'**{names} had no errors.**'))

            content.extend(protein_tables)

            condition_blocks.append(
                mo.vstack(
                    [
                        mo.md(f'## {condition}'),
                        mo.vstack(content)
                        if content
                        else mo.md('No results.'),
                    ]
                )
            )
        return mo.vstack(condition_blocks)

    _()


if __name__ == '__main__':
    app.run()
