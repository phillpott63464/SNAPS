from __future__ import annotations

from pathlib import Path
from typing import Optional

from refactor.SNAPSconfig import build_config
from refactor.SNAPSpipeline import run_pipeline

from modules.bmrb import process_bmrb, with_psipred_dir


@with_psipred_dir
def run_tripletm(
    bmrb_ids: list[int],
    DATA_DIR: Path,
    OUTPUT_DIR: Path,
    psipred_dir: Optional[Path] = None,
) -> dict:
    """
    No variables, used as reference for tripletm2.
    Very computationally expensive, but physically relevant, so theoretically should be the most accurate.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for bmrb_id in bmrb_ids:
        print(f'Processing BMRB ID: {bmrb_id}')
        output[bmrb_id] = {}
        data_dir = DATA_DIR / f'{bmrb_id}'
        output_dir = OUTPUT_DIR / f'{bmrb_id}/seq_link_test'

        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = process_bmrb(bmrb_id, data_dir, psipred_dir=psipred_dir)
        if files is None:
            continue
        base = {
            'obs_file': files['nef_file'],
            'pred_file': f"{files['shiftx2_file']}:shiftx2",
            'sequence_file': files['fasta_file'],
            'SS_num': True,
            'output_type': 'snaps',
            'strategy': 'triplet',
            'leave_one_out': True,
        }

        strategy = 'tripletm-basic'
        strat_files = {
            'result_file': output_dir
            / f'{bmrb_id}-test-result-{strategy}.txt',
            **files,
        }

        output[bmrb_id][strategy] = strat_files

        if strat_files['result_file'].exists():
            continue

        config = build_config(
            cli_overrides={
                'output_file': strat_files['result_file'],
                **base,
            }
        )
        run_pipeline(config)

    return output


@with_psipred_dir
def run_tripletm2(
    bmrb_ids: list[int],
    DATA_DIR: Path,
    OUTPUT_DIR: Path,
    cutoff_1: float = 0.1,
    cutoff_2: float = 0.5,
    psipred_dir: Optional[Path] = None,
) -> dict:
    """Run all configured SNAPS strategies, skipping existing result files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for bmrb_id in bmrb_ids:
        output[bmrb_id] = {}
        data_dir = DATA_DIR / f'{bmrb_id}'
        output_dir = OUTPUT_DIR / f'{bmrb_id}'

        data_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        files = process_bmrb(bmrb_id, data_dir, psipred_dir=psipred_dir)
        if files is None:
            continue
        strategy = f'{cutoff_1}_{cutoff_2}'

        strat_files = {
            'result_file': output_dir
            / f'{bmrb_id}-test-result-{strategy}.txt',
            **files,
        }

        base = {
            'obs_file': files['nef_file'],
            'pred_file': f"{files['shiftx2_file']}:shiftx2",
            'sequence_file': files['fasta_file'],
            'SS_num': True,
            'output_type': 'snaps',
            'strategy': 'triplet',
            'non_diagonal': True,
        }

        output[bmrb_id][strategy] = strat_files

        if strat_files['result_file'].exists():
            continue

        config = build_config(
            cli_overrides={
                'output_file': strat_files['result_file'],
                'cutoff_1': cutoff_1,
                'cutoff_2': cutoff_2,
                **base,
            }
        )
        run_pipeline(config)

    return output
