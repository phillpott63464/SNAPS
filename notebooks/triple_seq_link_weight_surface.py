import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import contextlib
    import os
    from pathlib import Path

    import optuna
    from modules.analysis import compare_results, plot_comparisons_big
    from modules.bmrb import (
        run_seq_link_test,
        run_test_set,
        run_triple_seq_link,
        with_psipred_dir,
    )
    from modules.mars import get_mars_data
    from modules.rbmrb import get_rbmrb
    from modules.results import assignment_table
    from optuna.visualization import plot_optimization_history
    from tqdm.auto import tqdm

    return (
        Path,
        assignment_table,
        compare_results,
        contextlib,
        get_mars_data,
        get_rbmrb,
        optuna,
        os,
        plot_comparisons_big,
        plot_optimization_history,
        run_seq_link_test,
        run_test_set,
        run_triple_seq_link,
        tqdm,
        with_psipred_dir,
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
    ] + get_rbmrb(DATA_DIR, 200)
    BMRB_IDS_TEST = get_rbmrb(DATA_DIR, 200, 'test')
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
    return BMRB_IDS, BMRB_IDS_TEST, BMRB_NAMES, DATA_DIR, OUTPUT_DIR


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
    return default_results, default_test_files


@app.cell
def _(default_test_files):
    print(default_test_files.keys())
    return


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    contextlib,
    default_results,
    optuna,
    os,
    plot_optimization_history,
    run_triple_seq_link,
    tqdm,
    with_psipred_dir,
):
    @with_psipred_dir
    def _(psipred_dir):
        def objective(trial):
            x = trial.suggest_float('CA', 0, 0.1)
            y = trial.suggest_float('CB', 0, 0.1)
            z = trial.suggest_float('C', 0, 0.1)
            weight = trial.suggest_float('weight', 1.0, 5.0)
            seq_links = trial.suggest_int('seq_links', 1, 3)

            seqs = {'CA': x, 'CB': y, 'C': z}
            print(seqs)
            files = run_triple_seq_link(
                bmrb_ids=BMRB_IDS,
                DATA_DIR=DATA_DIR,
                OUTPUT_DIR=OUTPUT_DIR / 'triple_seq',
                seqs=seqs,
                weight=weight,
                seq_links=seq_links,
                psipred_dir=psipred_dir,
            )
            results = assignment_table(files, BMRB_NAMES, 'candidate')
            comparison = compare_results(results, default_results)
            return comparison['err_rate_candidate'].mean()

        def silent_objective(trial):
            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(
                    f
                ):
                    return objective(trial)

        study = optuna.create_study(
            direction='minimize',
            study_name='Triple sequential links',
            storage=f'sqlite:///{OUTPUT_DIR}/tri_seq.db',
            load_if_exists=True,
        )

        n_trials = 200 - len(study.trials)

        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        with tqdm(total=n_trials, desc='Optuna') as pbar:

            def callback(study, trial):
                pbar.update(1)
                pbar.set_postfix(best=f'{study.best_value:.4f}')
                fig = plot_optimization_history(study)
                fig.show()

            study.optimize(
                silent_objective,
                n_trials=n_trials,
                callbacks=[callback],
                show_progress_bar=False,
            )

        return study

    study = _()
    return (study,)


@app.cell
def _(study):
    print(study.best_params, study.best_value)
    return


@app.cell
def _(
    BMRB_IDS_TEST,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    default_results,
    plot_comparisons_big,
    run_triple_seq_link,
    study,
):
    best_seqs = {
        'CA': study.best_params['CA'],
        'CB': study.best_params['CB'],
        'C': study.best_params['C'],
    }

    files = run_triple_seq_link(
        bmrb_ids=BMRB_IDS_TEST,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        seqs=best_seqs,
        seq_links=study.best_params['seq_links'],
        weight=study.best_params['weight'],
    )

    results = assignment_table(files, BMRB_NAMES, 'candidate (test set)')
    comparison = compare_results(results, default_results)
    plots = plot_comparisons_big(comparison)
    return plots, results


@app.cell
def _(mo, plots):
    mo.vstack([mo.as_html(plot) for plot in plots.values()])
    return


@app.cell
def _(plot_optimization_history, study):
    fig = plot_optimization_history(study)
    fig.show()
    return


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    compare_results,
    default_results,
    get_mars_data,
    plot_comparisons_big,
    results,
):
    mars_results = get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR)
    mars_default_comparison = compare_results(default_results, mars_results)

    mars_new_comparison = compare_results(results, mars_results)

    mars_plots = {
        **plot_comparisons_big(mars_default_comparison),
        **plot_comparisons_big(mars_new_comparison),
    }
    return (mars_plots,)


@app.cell
def _(mars_plots, mo):
    mo.vstack([mo.as_html(plot) for plot in mars_plots.values()])
    return


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    optuna,
    plot_comparisons_big,
    results,
    run_seq_link_test,
):
    single_weight_optimised_study = optuna.create_study(
        direction='minimize',
        study_name='Sequential link and weighting surface',
        storage=f'sqlite:///{OUTPUT_DIR}/seq_weight.db',
        load_if_exists=True,
    )

    single_weight_optimised_files = run_seq_link_test(
        bmrb_ids=BMRB_IDS,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        seq_range=[single_weight_optimised_study.best_params['threshold']],
        weight=single_weight_optimised_study.best_params['weight'],
    )

    single_weight_optimised_results = assignment_table(
        single_weight_optimised_files, BMRB_NAMES, 'single_weight_optimised'
    )
    single_weight_optimised_comparison = compare_results(
        results, single_weight_optimised_results
    )
    single_weight_optimised_plots = plot_comparisons_big(
        single_weight_optimised_comparison
    )
    return (single_weight_optimised_plots,)


@app.cell
def _(mo, single_weight_optimised_plots):
    mo.vstack(
        [
            mo.md(
                """# This only works if you've run seq_link_weight_surface first"""
            )
        ]
        + [mo.as_html(plot) for plot in single_weight_optimised_plots.values()]
    )
    return


if __name__ == "__main__":
    app.run()
