import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import re
    import subprocess
    import sys
    import urllib.error
    import urllib.request
    from pathlib import Path
    from refactor.SNAPSpipeline import run_pipeline
    from refactor.SNAPSconfig import build_config

    return Path, build_config, re, run_pipeline, subprocess, sys, urllib


@app.cell
def _(Path):
    BMRB_IDS = [5471, 4354, 4384, 4022, 4106, 4457, 4402, 4341, 547, 4082, 4136]
    DATA_DIR = Path(__file__).resolve().parent / "mars_test_set"
    OUTPUT_DIR = Path(__file__).resolve().parent / "mars_test_result"
    return BMRB_IDS, DATA_DIR, OUTPUT_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Download Test Data from BMRB for Mars Test Dataset

    ## Their Results:

    | Protein                             | Chemical shifts used | Condition I: assignments / reliable / errors | Condition II: assignments / reliable / errors | Condition III: assignments / reliable / errors |
    | ----------------------------------- | -------------------- | -------------------------------------------: | --------------------------------------------: | ---------------------------------------------: |
    | Malate synthase G                   | C′, Cα, Cβ           |                                652 / 639 / 0 |                                 652 / 639 / 0 |                                  651 / 623 / 0 |
    | Malate synthase G                   | Cα, Cβ*              |                                500 / 207 / 0 |                                 639 / 584 / 2 |                                  622 / 511 / 0 |
    | Maltose binding protein             | Cα, Cβ               |                                323 / 303 / 0 |                                 333 / 324 / 0 |                                  330 / 313 / 1 |
    | Rous Sarcoma Virus capsid           | C′, Cα, Cβ           |                                214 / 205 / 0 |                                 218 / 207 / 0 |                                  218 / 199 / 0 |
    | Human carbonic anhydrase I          | C′, Cα, Cβ           |                                242 / 235 / 0 |                                 242 / 237 / 0 |                                  242 / 225 / 0 |
    | N-terminal domain of enzyme I (EIN) | Cα, Cβ               |                                246 / 232 / 0 |                                 246 / 246 / 0 |                                  248 / 245 / 0 |
    | E-cadherin domains II and III       | Cα, Cβ               |                                 116 / 77 / 1 |                                 134 / 102 / 0 |                                   136 / 70 / 1 |
    | Human prion protein                 | Cα, Cβ               |                                138 / 103 / 0 |                                 155 / 127 / 0 |                                  154 / 118 / 0 |
    | Superoxide dismutase                | C′, Cα, Cβ           |                                112 / 101 / 0 |                                 112 / 104 / 0 |                                  111 / 100 / 0 |
    | Superoxide dismutase                | Cα, Cβ               |                                111 / 101 / 0 |                                 112 / 104 / 0 |                                  112 / 103 / 0 |
    | Calmodulin/M13                      | Cα, C′               |                                  97 / 37 / 0 |                                 144 / 142 / 0 |                                  136 / 119 / 0 |
    | Profilin                            | Cα, Cβ               |                                130 / 132 / 2 |                                 132 / 132 / 0 |                                  132 / 123 / 0 |
    | E. coli EmrE                        | C′, Cα, Cβ           |                                  61 / 35 / 0 |                                   70 / 58 / 0 |                                    64 / 50 / 0 |
    | Human ubiquitin                     | Cα, Cβ               |                                  72 / 72 / 0 |                                   72 / 72 / 0 |                                    72 / 70 / 0 |
    | Human ubiquitin                     | Cα only              |                                   32 / 9 / 0 |                                   58 / 18 / 0 |                                     58 / 9 / 0 |
    | Z domain                            | Cα, Cβ, Hα†          |                                  65 / 65 / 0 |                                             — |                                              — |
    | Z domain                            | Cα, Cβ‡              |                                  57 / 34 / 0 |                                             — |                                              — |
    | Tir110                              | C′, Cα, Cβ§          |                                  91 / 80 / 0 |                                             — |                                              — |

    Our goal: match their results, but in a more modern, FOSS method.
    """)
    return


@app.cell
def _(DATA_DIR, Path, re, subprocess, sys, urllib):
    # Download functions

    def download(url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": "Python"})
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 64):
                output.write(chunk)

    def download_first_pdb_for_bmrb(bmrb_id: int) -> bool:
        output_file = DATA_DIR / f"{bmrb_id}.pdb"
        tmp_file = DATA_DIR / f"{bmrb_id}.pdb.tmp"
        summary_url = (
            "https://bmrb.io/data_library/summary/index.php"
            f"?bmrbId={bmrb_id}"
        )

        try:
            request = urllib.request.Request(summary_url, headers={"User-Agent": "Python"})
            with urllib.request.urlopen(request) as response:
                html = response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, OSError) as exc:
            print(f"Could not retrieve BMRB {bmrb_id} summary: {exc}", file=sys.stderr)
            return False

        pdb_ids = re.findall(r"pdb-id='([0-9A-Za-z]{4})", html)

        for pdb_id in pdb_ids:
            pdb_id = pdb_id.lower()

            try:
                download(f"https://files.rcsb.org/download/{pdb_id}.pdb", tmp_file)

                with tmp_file.open("r", errors="replace") as pdb:
                    usable = any(
                        line.startswith(("ATOM", "HEADER", "TITLE"))
                        for line in pdb
                    )

                if usable:
                    tmp_file.replace(output_file)
                    print(
                        f"Downloaded PDB {pdb_id} linked to BMRB "
                        f"{bmrb_id} → {output_file}"
                    )
                    return True
            except (urllib.error.URLError, OSError):
                pass
            finally:
                tmp_file.unlink(missing_ok=True)

        print(
            f"No linked structure available in .pdb format for BMRB {bmrb_id}",
            file=sys.stderr,
        )
        return False

    def run_download_pipeline(commands: list[list[str]], output_file: Path) -> None:
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

            return_codes = [process.wait() for process in processes]
            for code, command in zip(return_codes, commands):
                if code != 0:
                    raise subprocess.CalledProcessError(code, command)

        except Exception:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            raise


    return download, download_first_pdb_for_bmrb, run_download_pipeline


@app.cell
def _(
    DATA_DIR,
    download,
    download_first_pdb_for_bmrb,
    run_download_pipeline,
    subprocess,
    sys,
    urllib,
):
    def process_bmrb(bmrb_id: int) -> None:
        print(bmrb_id)
        star_file = DATA_DIR / f"{bmrb_id}.str"

        try:
            download(
                f"https://bmrb.io/rest/bmrb/{bmrb_id}/nmr-star3",
                star_file,
            )

            run_download_pipeline(
                [
                    ["nef", "header"],
                    ["nef", "nmrstar", "import", "sequence", str(star_file)],
                    ["nef", "nmrstar", "import", "shifts", str(star_file)],
                ],
                DATA_DIR / f"{bmrb_id}.nef",
            )

            download_first_pdb_for_bmrb(bmrb_id)

            run_download_pipeline(
                [
                    ["nef", "header"],
                    ["nef", "nmrstar", "import", "sequence", str(star_file)],
                    [
                        "nef",
                        "shiftx2",
                        "import",
                        "shifts",
                        str(DATA_DIR / f"{bmrb_id}.pdb"),
                    ],
                ],
                DATA_DIR / f"{bmrb_id}-shiftx2.nef",
            )

        except (urllib.error.URLError, OSError, subprocess.CalledProcessError) as exc:
            print(f"Failed to process BMRB {bmrb_id}: {exc}", file=sys.stderr)

    return (process_bmrb,)


@app.cell
def _(
    BMRB_IDS,
    DATA_DIR,
    OUTPUT_DIR,
    build_config,
    process_bmrb,
    run_pipeline,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for bmrb_id in BMRB_IDS:
        process_bmrb(bmrb_id)

        config = build_config(
            cli_overrides = {
                'obs_file': DATA_DIR / f'{bmrb_id}.nef',
                'pred_file': DATA_DIR / f'{bmrb_id}-shiftx2.nef:shiftx2',
                'output_file': OUTPUT_DIR / f'{bmrb_id}-test-result.nef',
                'output_type': 'nef',
            }    
        )
        run_pipeline(config)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Comparing SNAPS

    Files are id.nef and id-shiftx2.nef

    Additionally, id.pdb (probably irrelevant unless we swap to local shiftx2/ucbshift/sparta) and id.str for nmrstar

    Requirements:
    - Run snapspipeline
    - Save results in OUTPUT_DIR
    - Figure out a way to import the real assignment
    - compare each result to the real assignment
    - And should the data be kept as in memory files only and just the outputs written? That's brittle. But it's also much cleaner.
    """)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
