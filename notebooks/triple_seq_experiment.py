import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Does varying CA/CB/C with the other two static have a significant effect?

    Results: Yeah but quite small once you hone in properly
    """
    )


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    from modules.analysis import compare_results, plot_comparisons
    from modules.bmrb import run_test_set, run_triple_seq_link_list
    from modules.results import assignment_table

    return (
        Path,
        assignment_table,
        compare_results,
        plot_comparisons,
        plt,
        run_test_set,
        run_triple_seq_link_list,
    )


@app.cell
def _(Path):
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
    run_test_set,
):
    default_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    default_results = assignment_table(
        default_test_files, BMRB_NAMES, 'default (1.0/0.2 all)'
    )
    default_results = default_results[
        default_results['condition'] == 'triplet'
    ]
    return (default_results,)


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    default_results,
    plot_comparisons,
    run_triple_seq_link_list,
):
    cb_seqs = []
    ca_seqs = []
    c_seqs = []
    for x in range(10):
        cb_seqs += [
            {
                'CA': 0.05,
                'CB': x / 100,
                'C': 0.05,
            }
        ]
        ca_seqs += [
            {
                'CA': x / 100,
                'CB': 0.05,
                'C': 0.05,
            }
        ]
        c_seqs += [
            {
                'CA': 0.05,
                'CB': 0.05,
                'C': x / 100,
            }
        ]

    print('running test set')
    cb_files = run_triple_seq_link_list(
        BMRB_IDS, DATA_DIR, OUTPUT_DIR / 'triple_seq', cb_seqs, 2
    )
    ca_files = run_triple_seq_link_list(
        BMRB_IDS, DATA_DIR, OUTPUT_DIR / 'triple_seq', ca_seqs, 2
    )
    c_files = run_triple_seq_link_list(
        BMRB_IDS, DATA_DIR, OUTPUT_DIR / 'triple_seq', c_seqs, 2
    )

    print('creating assignment tables')
    cb_seq_link_test_result = assignment_table(
        cb_files,
        BMRB_NAMES,
        'sequential link threshold',
    )
    ca_seq_link_test_result = assignment_table(
        ca_files,
        BMRB_NAMES,
        'sequential link threshold',
    )
    c_seq_link_test_result = assignment_table(
        c_files,
        BMRB_NAMES,
        'sequential link threshold',
    )

    print('Creating comparisons')
    cb_comparison = compare_results(cb_seq_link_test_result, default_results)
    ca_comparison = compare_results(ca_seq_link_test_result, default_results)
    c_comparison = compare_results(c_seq_link_test_result, default_results)

    print('Creating plots')
    cb_plots = plot_comparisons(cb_comparison)
    ca_plots = plot_comparisons(ca_comparison)
    c_plots = plot_comparisons(c_comparison)
    return (
        c_comparison,
        c_plots,
        ca_comparison,
        ca_plots,
        cb_comparison,
        cb_plots,
    )


@app.cell
def _(cb_plots, mo):
    mo.vstack([mo.as_html(plot) for plot in cb_plots.values()])


@app.cell
def _(ca_plots, mo):
    mo.vstack([mo.as_html(plot) for plot in ca_plots.values()])


@app.cell
def _(c_plots, mo):
    mo.vstack([mo.as_html(plot) for plot in c_plots.values()])


@app.cell
def _(plt):
    # weighting_results = a pandas dataframe
    import json

    def make_graphs(comparison, val):
        graph_vals = {'error': {}, 'reliable': {}, 'correct': {}}

        for condition in comparison['condition'].unique():
            # condition == float
            weighting_slice = comparison[
                comparison['condition'] == condition
            ]   # Only the condition we want
            error_average = weighting_slice[
                'err_rate_sequential link threshold'
            ].mean()
            reliable_average = weighting_slice[
                'reliable_frac_sequential link threshold'
            ].mean()
            correct_average = weighting_slice[
                'correct_frac_sequential link threshold'
            ].mean()

            graph_vals['error'][condition] = error_average
            graph_vals['reliable'][condition] = reliable_average
            graph_vals['correct'][condition] = correct_average

        weighting_graphs = {}

        for graph, vals in graph_vals.items():
            print(vals)
            conditions = sorted(vals)
            # Sort conditions so the curve is drawn in order
            averages = [vals[condition] for condition in conditions]
            conditions = sorted(json.loads(key)[val] for key in vals)

            fig, ax = plt.subplots()
            ax.scatter(conditions, averages)
            ax.plot(conditions, averages)

            ax.set_xlabel(f'sequential link {val} value')
            plt.xticks(rotation=45)
            ax.set_ylabel(f'Mean {graph} fraction')
            ax.set_title(f'{graph.capitalize()} vs. sequential link value')
            ax.grid(True, alpha=0.3)

            weighting_graphs[graph] = fig
        return weighting_graphs

    return (make_graphs,)


@app.cell
def _(cb_comparison, make_graphs, mo):
    mo.vstack(
        [
            mo.as_html(plot)
            for plot in make_graphs(cb_comparison, 'CB').values()
        ]
    )


@app.cell
def _(ca_comparison, make_graphs, mo):
    mo.vstack(
        [
            mo.as_html(plot)
            for plot in make_graphs(ca_comparison, 'CA').values()
        ]
    )


@app.cell
def _(c_comparison, make_graphs, mo):
    mo.vstack(
        [mo.as_html(plot) for plot in make_graphs(c_comparison, 'C').values()]
    )


if __name__ == '__main__':
    app.run()
