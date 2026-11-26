"""Download BMRB test inputs and run the SNAPS test pipeline."""

from __future__ import annotations

import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

# TODO: Have these return filepaths as like, dicts of paths and then have the processing take the dicts instead of rebuilding the paths constantly
# TODO: Also clean this up the folders have thousands of files in them. Maybe {bmrb_id}/files and {bmrb_id}/mars/files (because mars has huge output quantities too)
from functools import wraps
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import requests
from pynmrstar import Entry
from refactor.SNAPSconfig import build_config
from refactor.SNAPSpipeline import run_pipeline


def with_psipred_dir(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        psipred_dir = kwargs.pop('psipred_dir', None)
        if psipred_dir is not None:
            return func(*args, psipred_dir=psipred_dir, **kwargs)

        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            archive = temp_dir / 'psipred.tar.gz'

            context = ssl._create_unverified_context()

            with urlopen(
                'https://bioinf.cs.ucl.ac.uk/downloads/psipred/psipred.4.02.tar.gz',
                context=context,
            ) as response:
                archive.write_bytes(response.read())

            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(temp_dir, filter='data')

            return func(
                *args,
                psipred_dir=temp_dir / 'psipred',
                **kwargs,
            )

    return wrapper


def download(url: str, destination: Path) -> None:
    """Download *url* unless the destination already exists."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'Python'})
    with urllib.request.urlopen(request) as response, destination.open(
        'wb'
    ) as output:
        while chunk := response.read(1024 * 64):
            output.write(chunk)


def inspect_nmrstar(path):
    entry = Entry.from_file(path)
    peak_frames = []

    for frame in entry.frame_list:
        if getattr(frame, 'category', '').lower() != 'spectral_peak_list':
            continue

        # In these BMRB files, peak coordinates may be embedded as text.
        try:
            text_data = frame.get_tag('_Spectral_peak_list.Text_data')
        except KeyError:
            text_data = None

        has_text_data = bool(
            text_data and str(text_data).strip() not in {'', '.'}
        )
        has_peak_loop = any(
            any(tag.lower().startswith('_spectral_peak.') for tag in loop.tags)
            for loop in frame.loops
        )

        peak_frames.append(
            {
                'name': frame.name,
                'has_peak_data': has_text_data or has_peak_loop,
            }
        )

    return peak_frames


def keep_nef_chain_a(file_path: Path) -> None:
    entry = Entry.from_file(str(file_path))

    found_sequence_loop = False

    for saveframe in entry.frame_list:
        for loop in saveframe.loops:
            tags = [
                tag.lstrip('_').lower().split('.')[-1] for tag in loop.tags
            ]

            # The molecular-system sequence loop defines the NEF chains.
            if 'chain_code' in tags and any(
                tag.endswith('sequence_code') for tag in tags
            ):
                found_sequence_loop = True

            chain_indices = [
                index
                for index, tag in enumerate(tags)
                if tag == 'chain_code' or tag.startswith('chain_code_')
            ]

            if not chain_indices:
                continue

            retained_rows = [
                row
                for row in loop.data
                if all(
                    str(row[index]).strip('\'"') == 'A'
                    for index in chain_indices
                )
            ]
            loop.clear_data()
            if retained_rows:
                loop.add_data(retained_rows)

    if not found_sequence_loop:
        raise ValueError(f'No NEF sequence loop found in {file_path}')

    temporary_file = file_path.with_suffix(f'{file_path.suffix}.tmp')
    entry.write_to_file(str(temporary_file))
    temporary_file.replace(file_path)


def _trim_star_to_chain_a(star_file: Path) -> None:
    """Keep only STAR data belonging to chain A."""
    entry = Entry.from_file(star_file)
    assembly_loop = None

    for saveframe in entry.frame_list:
        for loop in saveframe.loops:
            tags = [tag.lstrip('_').lower() for tag in loop.tags]
            if 'pdb_chain_id' in tags and 'id' in tags:
                assembly_loop = (loop, tags)
                break
        if assembly_loop is not None:
            break

    if assembly_loop is None:
        raise ValueError(
            f'STAR file has no entity assembly chain mapping: {star_file}'
        )

    loop, tags = assembly_loop
    chain_index = tags.index('pdb_chain_id')
    assembly_id_index = tags.index('id')
    entity_id_index = tags.index('entity_id') if 'entity_id' in tags else None
    asym_id_index = tags.index('asym_id') if 'asym_id' in tags else None

    chain_a_rows = [
        row for row in loop.data if str(row[chain_index]).strip('\'"') == 'A'
    ]
    if not chain_a_rows and asym_id_index is not None:
        chain_a_rows = [
            row
            for row in loop.data
            if str(row[asym_id_index]).strip('\'"') == 'A'
        ]
    # BMRB entries often leave both chain labels as '.', while NEF assigns
    # the first molecular assembly to chain A.
    if not chain_a_rows:
        chain_a_rows = loop.data[:1]
    if not chain_a_rows:
        raise ValueError(f'STAR file has no chain A: {star_file}')

    assembly_ids = {row[assembly_id_index] for row in chain_a_rows}
    entity_ids = (
        {row[entity_id_index] for row in chain_a_rows}
        if entity_id_index is not None
        else set()
    )
    asym_ids = (
        {row[asym_id_index] for row in chain_a_rows}
        if asym_id_index is not None
        else set()
    )

    for saveframe in entry.frame_list:
        for loop in saveframe.loops:
            tags = [tag.lstrip('_').lower() for tag in loop.tags]
            filters = []
            if 'entity_assembly_id' in tags:
                index = tags.index('entity_assembly_id')
                filters.append(
                    lambda row, index=index: row[index] in assembly_ids
                )
            if 'entity_id' in tags and entity_ids:
                index = tags.index('entity_id')
                filters.append(
                    lambda row, index=index: row[index] in entity_ids
                )
            if 'entity_assembly_asym_id' in tags and asym_ids:
                index = tags.index('entity_assembly_asym_id')
                filters.append(lambda row, index=index: row[index] in asym_ids)
            if filters:
                retained_rows = [
                    row
                    for row in loop.data
                    if all(predicate(row) for predicate in filters)
                ]
                loop.clear_data()
                if retained_rows:
                    loop.add_data(retained_rows)

    temporary_file = star_file.with_suffix(f'{star_file.suffix}.tmp')
    entry.write_to_file(temporary_file)
    temporary_file.replace(star_file)


def _run_download_pipeline(
    commands: list[list[str]], output_file: Path, cwd: Path
) -> None:
    if output_file.exists():
        return

    # TODO: you can use nef as a library this is silly and brittle

    processes = []
    previous_stdout = None
    try:
        for command in commands:
            process = subprocess.Popen(
                command,
                stdin=previous_stdout,
                stdout=subprocess.PIPE,
                cwd=cwd,
            )
            if previous_stdout is not None:
                previous_stdout.close()
            previous_stdout = process.stdout
            processes.append(process)

        assert previous_stdout is not None
        with output_file.open('wb') as output:
            for chunk in iter(lambda: previous_stdout.read(1024 * 64), b''):
                output.write(chunk)

        for process, command in zip(processes, commands):
            if process.wait() != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, command
                )
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise


def download_first_pdb_for_bmrb(
    bmrb_id: int, data_dir: Path, output_file: Path
) -> bool:
    """Download the first usable PDB structure linked from a BMRB entry."""
    if output_file.exists():
        return True

    tmp_file = data_dir / f'{bmrb_id}.pdb.tmp'
    summary_url = (
        f'https://bmrb.io/data_library/summary/index.php?bmrbId={bmrb_id}'
    )
    try:
        request = urllib.request.Request(
            summary_url, headers={'User-Agent': 'Python'}
        )
        with urllib.request.urlopen(request) as response:
            html = response.read().decode('utf-8', errors='replace')
    except (urllib.error.URLError, OSError) as exc:
        print(
            f'Could not retrieve BMRB {bmrb_id} summary: {exc}',
            file=sys.stderr,
        )
        return False

    for pdb_id in re.findall(r"pdb-id='([0-9A-Za-z]{4})", html):
        try:
            download(
                f'https://files.rcsb.org/download/{pdb_id.lower()}.pdb',
                tmp_file,
            )
            with tmp_file.open('r', errors='replace') as pdb:
                usable = any(
                    line.startswith(('ATOM', 'HEADER', 'TITLE'))
                    for line in pdb
                )
            if usable:
                tmp_file.replace(output_file)
                print(
                    f'Downloaded PDB {pdb_id} linked to BMRB {bmrb_id} -> {output_file}'
                )
                return True
        except (urllib.error.URLError, OSError):
            continue
        finally:
            tmp_file.unlink(missing_ok=True)

    print(
        f'No linked structure available in .pdb format for BMRB {bmrb_id}',
        file=sys.stderr,
    )
    return False


@with_psipred_dir
def process_bmrb(
    bmrb_id: int, data_dir: Path, psipred_dir: Optional[Path] = None
) -> dict:
    def _failed(files):
        with open(files['break_file'], 'w') as f:
            f.write('Gibberish')

    """Create the NEF, predicted-shift, PDB, and FASTA inputs for one BMRB entry."""
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {
        'star_file': data_dir / f'{bmrb_id}.str',
        'nef_file': data_dir / f'{bmrb_id}.nef',
        'shiftx2_file': data_dir / f'{bmrb_id}-shiftx2.nef',
        'fasta_file': data_dir / f'{bmrb_id}.fasta',
        'pdb_file': data_dir / f'{bmrb_id}.pdb',
        'psipred_file': data_dir / f'{bmrb_id}_psipred.tab',
        'break_file': data_dir / f'{bmrb_id}_break',
    }
    if files['break_file'].exists():
        return None
    if files['nef_file'].exists() == False:
        try:
            _run_download_pipeline(
                [
                    ['nef', 'header'],
                    [
                        'nef',
                        'nmrstar',
                        'import',
                        'project',
                        str(bmrb_id),
                    ],
                ],
                files['nef_file'],
                data_dir,
            )
            keep_nef_chain_a(files['nef_file'])
        except:
            if files['nef_file'].exists():
                os.remove(files['nef_file'])
            return _failed(files)
        if files['nef_file'].exists() == False:
            return _failed(files)

    if files['pdb_file'].exists() == False:
        download_first_pdb_for_bmrb(bmrb_id, data_dir, files['pdb_file'])
        if files['pdb_file'].exists() == False:
            return _failed(files)
            # No pdb file on server, skip

    if files['shiftx2_file'].exists() == False:
        try:
            _run_download_pipeline(
                [
                    ['nef', 'header'],
                    [
                        'nef',
                        'stream',
                        str(files['nef_file']),
                    ],
                    [
                        'nef',
                        'shiftx2',
                        'import',
                        'shifts',
                        str(data_dir / f'{bmrb_id}.pdb'),
                    ],
                ],
                files['shiftx2_file'],
                data_dir,
            )
        except:
            if files['shiftx2_file'].exists():
                os.remove(files['shiftx2_file'])

            return _failed(files)

    with TemporaryDirectory() as tmp_dir:
        if (data_dir / f'{bmrb_id}.inp').exists() == False:
            _run_download_pipeline(
                [
                    ['nef', 'stream', f'{bmrb_id}.nef'],
                    ['nef', 'entry', 'rename', f'{bmrb_id}'],
                    [
                        'nef',
                        'fasta',
                        'export',
                        'sequence',
                        '--force',
                    ],  # Must use nef fasta not mars because mars export sequence breaks with 2+ chains
                    ['nef', 'mars', 'export', 'shifts', '--force'],
                    [
                        'nef',
                        'mars',
                        'export',
                        'input',
                        '--force',
                    ],  # Exports to bmrb_id.inp
                ],
                Path(tmp_dir) / 'abba',  # Discard pipe output
                data_dir,
            )

            # Replace mars.inp defaults
            input_file = data_dir / f'{bmrb_id}.inp'
            input_text = input_file.read_text()

            updates = {
                'cutoffCA': '0.2',
                'cutoffCB': '0.4',
                'cutoffCO': '0.15',
            }

            for key, value in updates.items():
                pattern = rf'^(\s*{key}:\s*)\S+'
                input_text, count = re.subn(
                    pattern,
                    lambda match: match.group(1) + value,
                    input_text,
                    count=1,
                    flags=re.MULTILINE,
                )
                if count != 1:
                    raise ValueError(
                        f'Expected one {key} setting; found {count}'
                    )

            input_file.write_text(input_text)

        if files['psipred_file'].exists() == False:
            shutil.copy(files['fasta_file'], f'{psipred_dir}/{bmrb_id}.fasta')
            script = psipred_dir / 'runpsipred_single'
            result = subprocess.run(
                [str(script), f'{bmrb_id}.fasta'],
                cwd=psipred_dir,
                text=True,
                capture_output=True,
            )

            walk_tree(psipred_dir)
            shutil.copy(
                f'{psipred_dir}/{bmrb_id}.horiz', files['psipred_file']
            )

        return files


@with_psipred_dir
def run_test_set(
    bmrb_ids: list[int],
    DATA_DIR: Path,
    OUTPUT_DIR: Path,
    psipred_dir: Optional[Path] = None,
) -> dict:
    """Run all configured SNAPS strategies, skipping existing result files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for bmrb_id in bmrb_ids:
        print(bmrb_id)
        output[bmrb_id] = {}
        data_dir = DATA_DIR / f'{bmrb_id}'
        output_dir = OUTPUT_DIR / f'{bmrb_id}'

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
        }

        for strategy in ('triplet', 'direct', 'iterative'):
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
                    'strategy': strategy,
                    'output_file': strat_files['result_file'],
                    **base,
                }
            )
            run_pipeline(config)

    return output


API_BASE = 'https://bioinf.cs.ucl.ac.uk/psipred/api/'


def download_psipred(
    fasta_file,
    email,
    output_dir,
    *,
    interval=30,
    timeout=60,
    max_wait=None,
):
    """Submit one FASTA sequence to PSIPRED and download its result files.

    Returns the directory containing the downloaded files.
    `interval` is in seconds; the PSIPRED example asks clients to wait
    at least 30 seconds between status checks.
    """
    fasta_file = Path(fasta_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if interval < 30:
        raise ValueError('Use an interval of at least 30 seconds.')

    # Keep the submission name within the server's documented 64-character limit.
    submission_name = fasta_file.stem[:64]
    result_dir = output_dir / f'{fasta_file.name} output'
    result_dir.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        # Submit the FASTA.
        with fasta_file.open('rb') as fasta:
            response = session.post(
                urljoin(API_BASE, 'submission.json'),
                data={
                    'job': 'psipred',
                    'submission_name': submission_name,
                    'email': email,
                },
                files={'input_data': (fasta_file.name, fasta)},
                headers={'Accept': 'application/json'},
                timeout=timeout,
            )

        response.raise_for_status()
        try:
            submission = response.json()
        except requests.exceptions.JSONDecodeError as exc:
            raise RuntimeError(
                'PSIPRED returned a non-JSON response to the submission '
                f'(HTTP {response.status_code}, '
                f"content type {response.headers.get('Content-Type')!r}):\n"
                f'{response.text[:1000]}'
            ) from exc

        job_id = submission.get('UUID')
        if not job_id:
            raise RuntimeError(
                f'PSIPRED did not provide a job UUID. Server response: {submission}'
            )

        # Poll until the job finishes, or until max_wait is reached.
        status_url = urljoin(API_BASE, f'submission/{job_id}')
        start = time.monotonic()

        while True:
            time.sleep(interval)

            response = session.get(
                status_url,
                params={'format': 'json'},
                headers={'Accept': 'application/json'},
                timeout=timeout,
            )
            response.raise_for_status()
            try:
                status = response.json()
            except requests.exceptions.JSONDecodeError as exc:
                raise RuntimeError(
                    'PSIPRED returned a non-JSON response while checking job '
                    f'{job_id} (HTTP {response.status_code}):\n'
                    f'{response.text[:1000]}'
                ) from exc

            state = status.get('state')
            if state == 'Complete':
                break
            if state == 'Error':
                raise RuntimeError(f'PSIPRED job {job_id} failed: {status}')

            print(f"PSIPRED job state: {state or 'unknown'}")

            if max_wait is not None and time.monotonic() - start >= max_wait:
                raise TimeoutError(
                    f'PSIPRED job {job_id} did not finish within {max_wait} seconds.'
                )

        # Download each result listed by the completed job.
        try:
            results = status['submissions'][0]['results']
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f'Completed job response did not contain result paths: {status}'
            ) from exc

        for result in results:
            data_path = result['data_path']
            result_url = urljoin(API_BASE, data_path.lstrip('/'))
            response = session.get(result_url, timeout=timeout)
            response.raise_for_status()

            filename = Path(urlparse(result_url).path).name
            if not filename:
                filename = 'psipred_result'
            (result_dir / filename).write_bytes(response.content)


@with_psipred_dir
def run_mars_set(
    bmrb_ids: list[int], DATA_DIR: Path, psipred_dir: Optional[Path] = None
) -> dict:
    output = {}
    output[
        'II'
    ] = (
        {}
    )   # Basically just defining a single condition at the moment so that everything is very simple
    for bmrb_id in bmrb_ids:
        print(f'{bmrb_id}')
        data_dir = DATA_DIR / str(bmrb_id)
        files = process_bmrb(bmrb_id, data_dir, psipred_dir=psipred_dir)
        if files is None:
            continue
        output['II'][bmrb_id] = files
        output['II'][bmrb_id] = {
            'assignment_AA': data_dir / 'assignment_AA.out',
            **output['II'][bmrb_id],
        }
        files = output['II'][bmrb_id].copy()

        # Find out how to tell what is available on path because if mars-podman isn't this could maybe not go well
        # Or include the dockerfiles in here?
        # But then not sure if we have podman installed either
        # Could also run a usual mars but not sure how to test native since I don't have native
        if (files['assignment_AA']).exists() == False:
            subprocess.run(
                ['mars-podman', 'run', f'{bmrb_id}.inp'], cwd=data_dir
            )

    return output


def walk_tree(path: Path):
    for root, dirs, files in os.walk(path):
        depth = root.count(os.sep)
        print('  ' * depth + os.path.basename(root) + '/')
        for name in files:
            print('  ' * (depth + 1) + name)


@with_psipred_dir
def run_weighting_test(
    bmrb_ids: list[int],
    DATA_DIR: Path,
    OUTPUT_DIR: Path,
    weighting_range: list[float],
    psipred_dir: Optional[Path] = None,
) -> dict:
    """Run all configured SNAPS strategies, skipping existing result files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for bmrb_id in bmrb_ids:
        output[bmrb_id] = {}
        data_dir = DATA_DIR / f'{bmrb_id}'
        output_dir = OUTPUT_DIR / f'{bmrb_id}/weighting_test'

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
        }

        for strategy in weighting_range:
            strategy = float(strategy)
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
                    'sequential_weight': strategy,
                    **base,
                }
            )
            run_pipeline(config)

    return output


@with_psipred_dir
def run_seq_link_test(
    bmrb_ids: list[int],
    DATA_DIR: Path,
    OUTPUT_DIR: Path,
    seq_range: list[float],
    weight: float = 1,
    psipred_dir: Optional[Path] = None,
) -> dict:
    """Run all configured SNAPS strategies, skipping existing result files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for bmrb_id in bmrb_ids:
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
        }

        for strategy in seq_range:
            strategy = float(strategy)
            strat_files = {
                'result_file': output_dir
                / f'{bmrb_id}-test-result-{weight}-{strategy}.txt',
                **files,
            }
            output[bmrb_id][strategy] = strat_files

            if strat_files['result_file'].exists():
                continue

            config = build_config(
                cli_overrides={
                    'output_file': strat_files['result_file'],
                    'seq_link_threshold': strategy,
                    'sequential_weight': weight,
                    **base,
                }
            )
            run_pipeline(config)

    return output


@with_psipred_dir
def run_triple_seq_link(
    bmrb_ids: list[int],
    DATA_DIR: Path,
    OUTPUT_DIR: Path,
    seqs: dict,
    weight: float = 1,
    seq_links: int = 3,
    psipred_dir: Optional[Path] = None,
) -> dict:
    """Run all configured SNAPS strategies, skipping existing result files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for bmrb_id in bmrb_ids:
        print(bmrb_id)
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
            'min_seq_links_required': seq_links,
        }

        for strategy in [''.join(x for x in str(seqs.values()))]:
            strategy = str(strategy)
            strat_files = {
                'result_file': output_dir
                / f'{bmrb_id}-test-result-{weight}-{strategy}.txt',
                **files,
            }
            output[bmrb_id][strategy] = strat_files

            if strat_files['result_file'].exists():
                continue

            config = build_config(
                cli_overrides={
                    'output_file': strat_files['result_file'],
                    'sequential_weight': weight,
                    'seq_link_threshold_ca': seqs['CA'],
                    'seq_link_threshold_cb': seqs['CB'],
                    'seq_link_threshold_co': seqs['C'],
                    **base,
                }
            )
            run_pipeline(config)

    return output


@with_psipred_dir
def run_triple_seq_link_list(
    bmrb_ids: list[int],
    DATA_DIR: Path,
    OUTPUT_DIR: Path,
    seqs_list: list[dict],
    weight: float = 1,
    psipred_dir: Optional[Path] = None,
) -> dict:
    """Run all configured SNAPS strategies, skipping existing result files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = {}
    for bmrb_id in bmrb_ids:
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
        }

        for seqs in seqs_list:
            strategy = str(seqs.values())
            strat_files = {
                'result_file': output_dir
                / f'{bmrb_id}-test-result-{weight}-{strategy}.txt',
                **files,
            }
            import json

            output[bmrb_id][json.dumps(seqs)] = strat_files

            if strat_files['result_file'].exists():
                continue

            config = build_config(
                cli_overrides={
                    'output_file': strat_files['result_file'],
                    'sequential_weight': weight,
                    'seq_link_threshold_ca': seqs['CA'],
                    'seq_link_threshold_cb': seqs['CB'],
                    'seq_link_threshold_co': seqs['C'],
                    **base,
                }
            )
            run_pipeline(config)

    return output
