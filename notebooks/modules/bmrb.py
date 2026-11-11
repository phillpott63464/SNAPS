"""Download BMRB test inputs and run the SNAPS test pipeline."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

from refactor.SNAPSpipeline import run_pipeline
from refactor.SNAPSconfig import build_config


def download(url: str, destination: Path) -> None:
    """Download *url* unless the destination already exists."""
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Python"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while chunk := response.read(1024 * 64):
            output.write(chunk)


def _run_download_pipeline(commands: list[list[str]], output_file: Path) -> None:
    if output_file.exists():
        return

    processes = []
    previous_stdout = None
    try:
        for command in commands:
            process = subprocess.Popen(
                command,
                stdin=previous_stdout,
                stdout=subprocess.PIPE,
            )
            if previous_stdout is not None:
                previous_stdout.close()
            previous_stdout = process.stdout
            processes.append(process)

        assert previous_stdout is not None
        with output_file.open("wb") as output:
            for chunk in iter(lambda: previous_stdout.read(1024 * 64), b""):
                output.write(chunk)

        for process, command in zip(processes, commands):
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, command)
    except Exception:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise


def download_first_pdb_for_bmrb(bmrb_id: int, data_dir: Path) -> bool:
    """Download the first usable PDB structure linked from a BMRB entry."""
    output_file = data_dir / f"{bmrb_id}.pdb"
    if output_file.exists():
        return True

    tmp_file = data_dir / f"{bmrb_id}.pdb.tmp"
    summary_url = f"https://bmrb.io/data_library/summary/index.php?bmrbId={bmrb_id}"
    try:
        request = urllib.request.Request(summary_url, headers={"User-Agent": "Python"})
        with urllib.request.urlopen(request) as response:
            html = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError) as exc:
        print(f"Could not retrieve BMRB {bmrb_id} summary: {exc}", file=sys.stderr)
        return False

    for pdb_id in re.findall(r"pdb-id='([0-9A-Za-z]{4})", html):
        try:
            download(f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb", tmp_file)
            with tmp_file.open("r", errors="replace") as pdb:
                usable = any(line.startswith(("ATOM", "HEADER", "TITLE")) for line in pdb)
            if usable:
                tmp_file.replace(output_file)
                print(f"Downloaded PDB {pdb_id} linked to BMRB {bmrb_id} -> {output_file}")
                return True
        except (urllib.error.URLError, OSError):
            continue
        finally:
            tmp_file.unlink(missing_ok=True)

    print(f"No linked structure available in .pdb format for BMRB {bmrb_id}", file=sys.stderr)
    return False


def process_bmrb(bmrb_id: int, data_dir: Path) -> None:
    """Create the NEF, predicted-shift, PDB, and FASTA inputs for one BMRB entry."""
    data_dir.mkdir(parents=True, exist_ok=True)
    star_file = data_dir / f"{bmrb_id}.str"
    try:
        download(f"https://bmrb.io/rest/bmrb/{bmrb_id}/nmr-star3", star_file)
        _run_download_pipeline(
            [["nef", "header"], ["nef", "nmrstar", "import", "sequence", str(star_file)],
             ["nef", "nmrstar", "import", "shifts", str(star_file)]],
            data_dir / f"{bmrb_id}.nef",
        )
        download_first_pdb_for_bmrb(bmrb_id, data_dir)
        _run_download_pipeline(
            [["nef", "header"], ["nef", "nmrstar", "import", "sequence", str(star_file)],
             ["nef", "shiftx2", "import", "shifts", str(data_dir / f"{bmrb_id}.pdb")]],
            data_dir / f"{bmrb_id}-shiftx2.nef",
        )
        fasta_file = data_dir / f"{bmrb_id}.fasta"
        if not fasta_file.exists():
            with (data_dir / f"{bmrb_id}.nef").open("rb") as nef_input:
                subprocess.run(["nef", "fasta", "export", "sequence", "--force"],
                               stdin=nef_input, cwd=data_dir, check=True)
            shutil.copyfile(data_dir / "nef.fasta", fasta_file)
    except (urllib.error.URLError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Failed to process BMRB {bmrb_id}: {exc}", file=sys.stderr)


def run_test_set(bmrb_ids: list[int], data_dir: Path, output_dir: Path) -> None:
    """Run all configured SNAPS strategies, skipping existing result files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for bmrb_id in bmrb_ids:
        process_bmrb(bmrb_id, data_dir)
        base = {
            "obs_file": data_dir / f"{bmrb_id}.nef",
            "pred_file": data_dir / f"{bmrb_id}-shiftx2.nef:shiftx2",
            "sequence_file": data_dir / f"{bmrb_id}.fasta",
            "SS_num": True,
            "output_type": "snaps",
        }
        for strategy in ("triplet", "direct", "iterative"):
            result_file = output_dir / f"{bmrb_id}-test-result-{strategy}.txt"
            if result_file.exists():
                continue
            config = build_config(cli_overrides={
                "strategy": strategy,
                "output_file": result_file,
                "hsqc_plot_file": output_dir / f"{bmrb_id}-test-result-{strategy}-hsqc.html",
                "strip_plot_file": output_dir / f"{bmrb_id}-test-result-{strategy}-strip.html",
                **base,
            })
            run_pipeline(config)
