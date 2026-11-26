import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


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
        with_psipred_dir,
    )
    from modules.mars import get_mars_data
    from modules.rbmrb import get_rbmrb
    from modules.results import assignment_table, read_test_result
    from modules.test_sets import run_tripletm, run_tripletm2
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
        read_test_result,
        run_tripletm,
        run_tripletm2,
        tqdm,
        with_psipred_dir,
    )


@app.cell
def _(Path):
    DATA_DIR = Path(__file__).resolve().parent / 'mars_test_set'
    OUTPUT_DIR = Path(__file__).resolve().parent / 'mars_test_result'
    BMRB_IDS = [
        # 5471,
        # 4354,
        # 4384,
        # 4022,
        # 4106,
        # 4457,
        # 4402,
        4341,
        547,
        4082,
        4136,
        5387,
    ]   # + get_rbmrb(DATA_DIR, 200)
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
def _(default_test_files):
    print(default_test_files.keys())


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    run_tripletm,
):
    default_test_files = run_tripletm(
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
def _(
    BMRB_IDS,
    DATA_DIR,
    OUTPUT_DIR,
    contextlib,
    default_test_files,
    optuna,
    os,
    plot_optimization_history,
    read_test_result,
    run_tripletm2,
    tqdm,
    with_psipred_dir,
):
    @with_psipred_dir
    def _(psipred_dir):
        def objective(trial):
            cutoff_2 = trial.suggest_float('cutoff_2', 0, 100)
            cutoff_1 = trial.suggest_float('cutoff_1', 0, cutoff_2)

            files = run_tripletm2(
                bmrb_ids=BMRB_IDS,
                DATA_DIR=DATA_DIR,
                OUTPUT_DIR=OUTPUT_DIR / 'tripletm',
                cutoff_1=cutoff_1,
                cutoff_2=cutoff_2,
                psipred_dir=psipred_dir,
            )
            error = 0
            for bmrb_id in BMRB_IDS:
                df = read_test_result(
                    files[bmrb_id][f'{cutoff_1}_{cutoff_2}']['result_file']
                )
                df = df[['Res_name', 'SS_name', 'Confidence']]

                odf = read_test_result(
                    default_test_files[bmrb_id]['tripletm-basic'][
                        'result_file'
                    ]
                )
                odf = odf[['Res_name', 'SS_name', 'Confidence']]

                for res_name in df['Res_name']:
                    row = df[df['Res_name'] == res_name]
                    orow = odf[odf['Res_name'] == res_name]

                    if row.empty or orow.empty:
                        continue

                    # Mismatches:
                    # same to same, +0
                    # High to Medium, +1
                    # High to Unreliable, +5
                    # Medium to High, +1
                    # Medium to Unreliable, +1
                    # Unreliable to High, +5
                    # Unreliable to Medium, +1

                    print(row['Confidence'].values[0])

                    if (
                        row['Confidence'].values[0]
                        == orow['Confidence'].values[0]
                    ):
                        error += 0
                    elif (
                        row['Confidence'].values[0] == 'High'
                        and orow['Confidence'].values[0] == 'Medium'
                    ):
                        error += 1
                    elif (
                        row['Confidence'].values[0] == 'High'
                        and orow['Confidence'].values[0] == 'Unreliable'
                    ):
                        error += 5
                    elif (
                        row['Confidence'].values[0] == 'Medium'
                        and orow['Confidence'].values[0] == 'High'
                        or row['Confidence'].values[0] == 'Medium'
                        and orow['Confidence'].values[0] == 'Unreliable'
                    ):
                        error += 1
                    elif (
                        row['Confidence'].values[0] == 'Unreliable'
                        and orow['Confidence'].values[0] == 'High'
                    ):
                        error += 5
                    elif (
                        row['Confidence'].values[0] == 'Unreliable'
                        and orow['Confidence'].values[0] == 'Medium'
                    ):
                        error += 1

            return error

        def silent_objective(trial):
            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(
                    f
                ):
                    return objective(trial)

        study = optuna.create_study(
            direction='minimize',
            study_name='Make tripletm2 be kinda the same as tripletm',
            storage=f'sqlite:///{OUTPUT_DIR}/tripletm.db',
            load_if_exists=True,
        )

        n_trials = 30 - len(study.trials)

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


@app.cell
def _(DATA_DIR, get_rbmrb):
    BMRB_IDS_TEST = get_rbmrb(DATA_DIR, 20, 'test')
    BMRB_NAMES_TEST = {}

    for key_2 in BMRB_IDS_TEST:
        BMRB_NAMES_TEST.setdefault(key_2, str(key_2))

    print(len(BMRB_IDS_TEST))
    print(len(BMRB_NAMES_TEST))

    for name in BMRB_NAMES_TEST:
        assert name in BMRB_IDS_TEST

    for val in BMRB_IDS_TEST:
        assert name in BMRB_NAMES_TEST
    return BMRB_IDS_TEST, BMRB_NAMES_TEST


@app.cell
def _(
    BMRB_IDS,
    BMRB_IDS_TEST,
    BMRB_NAMES_TEST,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    plot_comparisons_big,
    psipred_dir,
    run_tripletm,
    run_tripletm2,
    study,
):
    test_default_test_files = run_tripletm(
        bmrb_ids=BMRB_IDS_TEST, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    test_default_results = assignment_table(
        test_default_test_files, BMRB_NAMES_TEST, 'tripletm'
    )

    files = run_tripletm2(
        bmrb_ids=BMRB_IDS,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR / 'tripletm',
        cutoff_1=study.best_params['cutoff_1'],
        cutoff_2=study.best_params['cutoff_2'],
        psipred_dir=psipred_dir,
    )

    results = assignment_table(files, BMRB_NAMES_TEST, 'candidate (test set)')
    comparison = compare_results(results, test_default_results)
    plots = plot_comparisons_big(comparison)
    return comparison, plots, results


@app.cell
def _(mo, plots):
    mo.vstack([mo.as_html(plot) for plot in plots.values()])


@app.cell
def _(comparison):
    print(comparison['err_rate_candidate (test set)'].mean())
    print(comparison['err_rate_default (1.0/0.2 all)'].mean())


@app.cell
def _(plot_optimization_history, study):
    fig = plot_optimization_history(study)
    fig.show()


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


if __name__ == '__main__':
    app.run()
