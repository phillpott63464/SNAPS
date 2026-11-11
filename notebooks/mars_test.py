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
    import shutil

    return (
        Path,
        build_config,
        re,
        run_pipeline,
        shutil,
        subprocess,
        sys,
        urllib,
    )


@app.cell
def _(Path):
    BMRB_IDS = [5471, 4354, 4384, 4022, 4106, 4457, 4402, 4341, 547, 4082, 4136]
    BMRB_NAMES = {
        5471: "Malate synthase G",
        4354: "Maltose binding protein",
        4384: "Rous Sarcoma Virus capsid",
        4022: "Human carbonic anhydrase I",
        4106: "N-terminal domain of enzyme I (EIN)",
        4457: "E-cadherin domains II and III",
        4402: "Human prion protein",
        4341: "Superoxide dismutase",
        547: "Calmodulin/M13",
        4082: "Profilin",
        4136: "E. coli EmrE",
    }
    DATA_DIR = Path(__file__).resolve().parent / "mars_test_set"
    OUTPUT_DIR = Path(__file__).resolve().parent / "mars_test_result"
    return BMRB_IDS, BMRB_NAMES, DATA_DIR, OUTPUT_DIR


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    # Download Test Data from BMRB for Mars Test Dataset

    ## Their Results:

    | Protein                             | Chemical shifts used   | Condition I: assignments / reliable / errors   | Condition II: assignments / reliable / errors   | Condition III: assignments / reliable / errors   |
    |:------------------------------------|:-----------------------|:-----------------------------------------------|:------------------------------------------------|:-------------------------------------------------|
    | Malate synthase G                   | C′, Cα, Cβ             | 652 / 639 / 0                                  | 652 / 639 / 0                                   | 651 / 623 / 0                                    |
    | Malate synthase G                   | Cα, Cβ*                | 500 / 207 / 0                                  | 639 / 584 / 2                                   | 622 / 511 / 0                                    |
    | Maltose binding protein             | Cα, Cβ                 | 323 / 303 / 0                                  | 333 / 324 / 0                                   | 330 / 313 / 1                                    |
    | Rous Sarcoma Virus capsid           | C′, Cα, Cβ             | 214 / 205 / 0                                  | 218 / 207 / 0                                   | 218 / 199 / 0                                    |
    | Human carbonic anhydrase I          | C′, Cα, Cβ             | 242 / 235 / 0                                  | 242 / 237 / 0                                   | 242 / 225 / 0                                    |
    | N-terminal domain of enzyme I (EIN) | Cα, Cβ                 | 246 / 232 / 0                                  | 246 / 246 / 0                                   | 248 / 245 / 0                                    |
    | E-cadherin domains II and III       | Cα, Cβ                 | 116 / 77 / 1                                   | 134 / 102 / 0                                   | 136 / 70 / 1                                     |
    | Human prion protein                 | Cα, Cβ                 | 138 / 103 / 0                                  | 155 / 127 / 0                                   | 154 / 118 / 0                                    |
    | Superoxide dismutase                | C′, Cα, Cβ             | 112 / 101 / 0                                  | 112 / 104 / 0                                   | 111 / 100 / 0                                    |
    | Superoxide dismutase                | Cα, Cβ                 | 111 / 101 / 0                                  | 112 / 104 / 0                                   | 112 / 103 / 0                                    |
    | Calmodulin/M13                      | Cα, C′                 | 97 / 37 / 0                                    | 144 / 142 / 0                                   | 136 / 119 / 0                                    |
    | Profilin                            | Cα, Cβ                 | 130 / 132 / 2                                  | 132 / 132 / 0                                   | 132 / 123 / 0                                    |
    | E. coli EmrE                        | C′, Cα, Cβ             | 61 / 35 / 0                                    | 70 / 58 / 0                                     | 64 / 50 / 0                                      |
    | Human ubiquitin                     | Cα, Cβ                 | 72 / 72 / 0                                    | 72 / 72 / 0                                     | 72 / 70 / 0                                      |
    | Human ubiquitin                     | Cα only                | 32 / 9 / 0                                     | 58 / 18 / 0                                     | 58 / 9 / 0                                       |
    | Z domain                            | Cα, Cβ, Hα†            | 65 / 65 / 0                                    | —                                               | —                                                |
    | Z domain                            | Cα, Cβ‡                | 57 / 34 / 0                                    | —                                               | —                                                |
    | Tir110                              | C′, Cα, Cβ§            | 91 / 80 / 0                                    | —                                               | —                                                |


    Our goal: match their results, but in a more modern, FOSS method.
    """)
    return


@app.cell
def _(DATA_DIR, Path, re, subprocess, sys, urllib):
    # Download functions

    def download(url: str, destination: Path) -> None:
        if destination.exists():
            return
        request = urllib.request.Request(url, headers={"User-Agent": "Python"})
        with urllib.request.urlopen(request) as response, destination.open("wb") as output:
            while chunk := response.read(1024 * 64):
                output.write(chunk)

    def download_first_pdb_for_bmrb(bmrb_id: int) -> bool:
        output_file = DATA_DIR / f"{bmrb_id}.pdb"
        if output_file.exists():
            return True
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

        if output_file.exists():
            return True

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
    Path,
    download,
    download_first_pdb_for_bmrb,
    run_download_pipeline,
    shutil,
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

            if (DATA_DIR / f"{bmrb_id}.fasta").exists() == False:
                with Path(DATA_DIR / f"{bmrb_id}.nef").open("rb") as nef_input:
                    subprocess.run(
                        ["nef", "fasta", "export", "sequence", "--force"],
                        stdin=nef_input,
                        cwd=DATA_DIR,
                        check=True,
                    )

                shutil.copyfile(DATA_DIR / "nef.fasta", DATA_DIR / f"{bmrb_id}.fasta")

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

    strategies = ['triplet', 'direct', 'iterative']
    file_types = ['nef', 'txt']

    for bmrb_id in BMRB_IDS:
        process_bmrb(bmrb_id)

        base_overrides = {
            'obs_file': DATA_DIR / f'{bmrb_id}.nef',
            'pred_file': DATA_DIR / f'{bmrb_id}-shiftx2.nef:shiftx2',
            'sequence_file': DATA_DIR / f'{bmrb_id}.fasta',
            'SS_num': True,
            'output_type': 'snaps',
        }

        for strategy in strategies:
            if (OUTPUT_DIR / f'{bmrb_id}-test-result-{strategy}.txt').exists():
                    continue

            config = build_config(
                cli_overrides = {
                    'strategy': strategy,
                    'output_file': OUTPUT_DIR / f'{bmrb_id}-test-result-{strategy}.txt',
                    'hsqc_plot_file': OUTPUT_DIR / f'{bmrb_id}-test-result-{strategy}-hsqc.html',
                    'strip_plot_file': OUTPUT_DIR / f'{bmrb_id}-test-result-{strategy}-strip.html',
                    **base_overrides
                }
            )

            run_pipeline(config)

    stupid_var = True # Marimo problems - the analysis won't automatically run unless there's some direct in python link between the two blocks
    return (stupid_var,)


@app.cell(hide_code=True)
def _(mo, snaps_results):
    mo.md(rf"""
    # Comparing SNAPS

    Files are id.nef and id-shiftx2.nef

    Additionally, id.pdb (probably irrelevant unless we swap to local shiftx2/ucbshift/sparta) and id.str for nmrstar

    Requirements:
    - Run snapspipeline
    - Save results in OUTPUT_DIR
    - Figure out a way to import the real assignment
    - compare each result to the real assignment
    - And should the data be kept as in memory files only and just the outputs written? That's brittle. But it's also much cleaner.

    {mo.md(snaps_results.to_markdown(index=False))}
    """)
    return


@app.cell
def _(BMRB_IDS, BMRB_NAMES, OUTPUT_DIR, Path, stupid_var):
    # Create our results

    if stupid_var == True:
        pass

    import shlex

    import pandas as pd
    import pynmrstar

    LOOP_PREFIX = "_nefpls_assignments."


    # --------------------------------------------------------------------------- #
    # readers
    # --------------------------------------------------------------------------- #
    def read_test_result(path: Path) -> pd.DataFrame:
        if path.suffix == '.nef':
            return read_test_result_nef(path)
        else:
            return read_test_result_snaps(path)

    def read_test_result_snaps(path: Path) -> pd.DataFrame:
        """Parse the assignment table printed by SNAPS text output."""
        lines = Path(path).read_text().splitlines()

        required = {
            "Res_name",
            "Res_N",
            "SS_name",
            "Log_prob",
            "Confidence",
        }
        header_index = None
        headings = None

        for i, line in enumerate(lines):
            fields = shlex.split(line.strip())
            if required.issubset(fields):
                header_index = i
                headings = fields
                break

        if header_index is None:
            raise ValueError(f"No SNAPS assignment table found in {path}")

        rows = []
        for line in lines[header_index + 1 :]:
            s = line.strip()
            if not s:
                continue

            fields = shlex.split(s)
            if len(fields) != len(headings):
                # The table has ended; following lines are status output.
                if rows:
                    break
                continue

            rows.append(fields)

        if not rows:
            raise ValueError(f"No assignment rows found in {path}")

        df = pd.DataFrame(rows, columns=headings)

        numeric_columns = [
            "Res_N",
            "Log_prob",
            "Max_mismatch_m1",
            "Max_mismatch_p1",
            "Max_mismatch",
            "Num_good_links_m1",
            "Num_good_links_p1",
            "Num_good_links",
        ]
        for column in numeric_columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in ("Dummy_res", "Dummy_SS"):
            if column in df.columns:
                df[column] = df[column].str.lower().map(
                    {"true": True, "false": False}
                )

        # In the SNAPS table, SS_name is the spin-system label and Res_N
        # is the assigned residue number.
        df["spin_system"] = df["SS_name"]
        df["residue_number"] = df["Res_N"]

        AA3 = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU",
        "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
        "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
        "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
        "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }

        # After building df in read_test_result_snaps:
        df["assigned"] = ~df["Dummy_res"]
        df["chain_code"] = "A"  # SNAPS text output doesn't include the chain
        df["residue_name"] = df["Res_type"].str.upper().map(AA3).fillna(df["Res_type"])
        df["unassigned_sequence_code"] = df["SS_name"]
        df["sequence_code"] = df["Res_N"]
        df["spin_system"] = pd.to_numeric(
            df["SS_name"].str.extract(r"^\s*(-?\d+)")[0], errors="coerce"
        )
        df["residue_number"] = pd.to_numeric(df["Res_N"], errors="coerce")

        return df

    def read_test_result_nef(path: Path) -> pd.DataFrame:
        """Parse the _nefpls_assignments loop with a plain text parser.

        (The result file has no data_ header / sf_category, so pynmrstar can't be
        relied on. shlex handles quoted values such as ' 149'.)
        """
        headings, rows = [], []
        in_loop = in_data = False
        for line in Path(path).read_text().splitlines():
            s = line.strip()
            if s == "loop_":
                in_loop, in_data = True, False
                headings, rows = [], []
                continue
            if not in_loop:
                continue
            if s.startswith(LOOP_PREFIX):
                headings.append(s[len(LOOP_PREFIX):].split()[0])
            elif s == "stop_":
                if headings:
                    break
                in_loop = False
            elif s and headings:
                in_data = True
                rows.append(shlex.split(s))
        if not rows:
            raise ValueError(f"No {LOOP_PREFIX} loop found in {path}")

        df = pd.DataFrame(rows, columns=headings)
        df["assigned"] = df["assigned"].str.lower() == "true"
        df["merit"] = df["merit"].astype(float)
        df["spin_system"] = pd.to_numeric(df["unassigned_sequence_code"].str.strip(), errors="coerce")
        df["residue_number"] = pd.to_numeric(df["sequence_code"], errors="coerce")  # '.' -> NaN
        return df


    def _loop_to_df(loop: pynmrstar.Loop) -> pd.DataFrame:
        cols = [t.split(".")[-1] for t in loop.get_tag_names()]
        return pd.DataFrame(loop.data, columns=cols)


    def read_initial_nef(path: Path, shift_list: str = None):
        """Return (sequence_df, shifts_df) from the initial NEF file."""
        entry = pynmrstar.Entry.from_string(Path(path).read_text())

        seq = _loop_to_df(entry.get_saveframes_by_category("nef_molecular_system")[0]
                          .get_loop("nef_sequence"))
        seq["sequence_code"] = seq["sequence_code"].astype(int)

        frames = entry.get_saveframes_by_category("nef_chemical_shift_list")
        if shift_list:
            frames = [f for f in frames if f.name.endswith(shift_list)] or frames
        shifts = _loop_to_df(frames[0].get_loop("nef_chemical_shift"))
        shifts["sequence_code"] = shifts["sequence_code"].astype(int)
        return seq, shifts


    # --------------------------------------------------------------------------- #
    # comparison
    # --------------------------------------------------------------------------- #
    def compare(result: pd.DataFrame, seq: pd.DataFrame, shifts: pd.DataFrame, chain="A"):
        seq = seq[seq["chain_code"] == chain]
        shifts = shifts[shifts["chain_code"] == chain]
        true_name = dict(zip(seq["sequence_code"], seq["residue_name"]))
        residues_with_shifts = set(shifts["sequence_code"])

        assigned = result[result["assigned"] & (result["chain_code"] == chain)].copy()
        assigned["residue_number"] = assigned["residue_number"].astype(int)
        bad = assigned[
            assigned["spin_system"].isna()
            | assigned["residue_number"].isna()
        ]

        if not bad.empty:
            raise ValueError(
                "Assigned rows contain labels that cannot be compared as residue numbers:\n"
                + bad[
                    ["unassigned_sequence_code", "sequence_code",
                     "chain_code", "residue_name"]
                ].to_string(index=False)
            )
        assigned["spin_system"] = assigned["spin_system"].astype(int)

        assigned["correct"] = assigned["spin_system"] == assigned["residue_number"]
        # residue type of the spin system according to the initial assignment
        assigned["true_residue_name"] = assigned["spin_system"].map(true_name)
        assigned["type_match"] = assigned["true_residue_name"] == assigned["residue_name"]
        assigned["offset"] = assigned["spin_system"] - assigned["residue_number"]
        # does the initial NEF actually have any shifts for this residue?
        assigned["has_reference_shifts"] = assigned["residue_number"].isin(residues_with_shifts)

        unassigned = result[~result["assigned"]].copy()
        return assigned, unassigned, residues_with_shifts


    # Set this to the reliability cutoff used by your method.
    RELIABLE_MERIT = 0.01
    # This is not what MARS used but oh well, can TODO that later

    # Add protein names and shift descriptions, keyed by your BMRB IDs.
    # Use the same ID type as BMRB_IDS (e.g. integers or strings).
    PROTEIN_INFO = {
        # 12345: ("Malate synthase G", "C′, Cα, Cβ"),
    }

    # Configure the result-file locations for each condition.
    # Add II and III when those results are available.
    STRATEGIES = {
        "direct": lambda bmrb_id: OUTPUT_DIR / f"{bmrb_id}-test-result-direct.txt",
        "iterative": lambda bmrb_id: OUTPUT_DIR / f"{bmrb_id}-test-result-iterative.txt",
        "triplet": lambda bmrb_id: OUTPUT_DIR / f"{bmrb_id}-test-result-triplet.txt",
    }

    def assignment_counts(result, chain="A", label_to_residue=None):
        assigned = result.loc[
            result["assigned"]
            & result["chain_code"].eq(chain)
            & result["residue_name"].ne("PRO")
        ].copy()

        if "merit" in assigned.columns:
            reliable = assigned.loc[
                assigned["merit"].le(RELIABLE_MERIT)
            ].copy()
        elif "Confidence" in assigned.columns:
            reliable = assigned.loc[
                assigned["Confidence"].isin(["High", "Medium"])
            ].copy()
        else:
            raise ValueError("Result has neither merit nor Confidence")

        reference = reliable["spin_system"].copy()

        if label_to_residue is not None:
            labels = reliable["unassigned_sequence_code"].astype(str).str.strip()
            mapped = pd.to_numeric(labels.map(label_to_residue), errors="coerce")
            reference = mapped.combine_first(reference)

        predicted = pd.to_numeric(reliable["residue_number"], errors="coerce")
        reference = pd.to_numeric(reference, errors="coerce")

        evaluable = (
            reference.notna()
            & predicted.notna()
            & reference.mod(1).eq(0)
            & predicted.mod(1).eq(0)
        )

        errors = (
            int(reference[evaluable].ne(predicted[evaluable]).sum())
            if evaluable.all()
            else float("nan")
        )

        return {
            "assignments": int(len(assigned)),
            "reliable_assignments": int(len(reliable)),
            "errors": errors,
        }


    def assignment_table(
        bmrb_ids,
        conditions,
        protein_info,
        chain="A",
        label_maps=None,
    ):
        label_maps = label_maps or {}
        rows = []

        for bmrb_id in bmrb_ids:
            protein, shifts_used = protein_info.get(
                bmrb_id, (BMRB_NAMES[bmrb_id], "—")
            )

            for condition in (STRATEGIES):
                path_factory = conditions.get(condition)
                path = Path(path_factory(bmrb_id)) if path_factory else None

                counts = (
                    assignment_counts(
                        read_test_result(path),
                        chain=chain,
                        label_to_residue=label_maps.get(bmrb_id),
                    )
                    if path is not None and path.is_file()
                    else {
                        "assignments": float("nan"),
                        "reliable_assignments": float("nan"),
                        "errors": float("nan"),
                    }
                )

                rows.append({
                    "bmrb_id": bmrb_id,
                    "protein": protein,
                    "chemical_shifts_used": shifts_used,
                    "condition": condition,
                    **counts,
                })

        df = pd.DataFrame(rows)

        for column in ("assignments", "reliable_assignments"):
            df[column] = df[column].astype("Int64")

        return df

    snaps_results = assignment_table(
        BMRB_IDS,
        STRATEGIES,
        PROTEIN_INFO,
    )

    print(snaps_results)
    return pd, snaps_results


@app.cell
def _(pd):
    data = [
        ("Malate synthase G", "C′, Cα, Cβ",
         (652, 639, 0), (652, 639, 0), (651, 623, 0)),
        ("Malate synthase G", "Cα, Cβ*",
         (500, 207, 0), (639, 584, 2), (622, 511, 0)),
        ("Maltose binding protein", "Cα, Cβ",
         (323, 303, 0), (333, 324, 0), (330, 313, 1)),
        ("Rous Sarcoma Virus capsid", "C′, Cα, Cβ",
         (214, 205, 0), (218, 207, 0), (218, 199, 0)),
        ("Human carbonic anhydrase I", "C′, Cα, Cβ",
         (242, 235, 0), (242, 237, 0), (242, 225, 0)),
        ("N-terminal domain of enzyme I (EIN)", "Cα, Cβ",
         (246, 232, 0), (246, 246, 0), (248, 245, 0)),
        ("E-cadherin domains II and III", "Cα, Cβ",
         (116, 77, 1), (134, 102, 0), (136, 70, 1)),
        ("Human prion protein", "Cα, Cβ",
         (138, 103, 0), (155, 127, 0), (154, 118, 0)),
        ("Superoxide dismutase", "C′, Cα, Cβ",
         (112, 101, 0), (112, 104, 0), (111, 100, 0)),
        ("Superoxide dismutase", "Cα, Cβ",
         (111, 101, 0), (112, 104, 0), (112, 103, 0)),
        ("Calmodulin/M13", "Cα, C′",
         (97, 37, 0), (144, 142, 0), (136, 119, 0)),
        ("Profilin", "Cα, Cβ",
         (130, 132, 2), (132, 132, 0), (132, 123, 0)),
        ("E. coli EmrE", "C′, Cα, Cβ",
         (61, 35, 0), (70, 58, 0), (64, 50, 0)),
        ("Human ubiquitin", "Cα, Cβ",
         (72, 72, 0), (72, 72, 0), (72, 70, 0)),
        ("Human ubiquitin", "Cα only",
         (32, 9, 0), (58, 18, 0), (58, 9, 0)),
        ("Z domain", "Cα, Cβ, Hα†",
         (65, 65, 0), None, None),
        ("Z domain", "Cα, Cβ‡",
         (57, 34, 0), None, None),
        ("Tir110", "C′, Cα, Cβ§",
         (91, 80, 0), None, None),
    ]

    conditions = ["I", "III", "III"]
    rows = []

    for protein, shifts_used, *condition_values in data:
        for condition, values in zip(conditions, condition_values):
            assignments, reliable, errors = values if values is not None else (pd.NA,) * 3
            rows.append({
                "bmrb_id": None,  # Fill in IDs when available
                "protein": protein,
                "chemical_shifts_used": shifts_used,
                "condition": condition,
                "assignments": assignments,
                "reliable_assignments": reliable,
                "errors": errors,
            })

    mars_results = pd.DataFrame(rows)
    mars_results[["assignments", "reliable_assignments", "errors"]] = (
        mars_results[["assignments", "reliable_assignments", "errors"]].astype("Int64")
    )
    return (mars_results,)


@app.cell(hide_code=True)
def _(mars_results, mo, snaps_results):
    mo.md(rf"""
    {mo.md(snaps_results.to_markdown(index=False))}

    {mo.md(mars_results.to_markdown(index=False))}
    """)
    return


@app.cell
def _(mars_results, snaps_results):
    print('SNAPS')
    print(snaps_results)
    print('=========')
    print('MARS')
    print(mars_results)
    return


@app.cell(hide_code=True)
def _(m, mo):
    mo.md(rf"""
    {mo.md(m.to_markdown(index=False))}
    """)
    return


@app.cell
def _(mars_results, pd, snaps_results):
    def _():
        from scipy import stats
        import numpy as np

        num = ['assignments', 'reliable_assignments', 'errors']
        snaps, mars = snaps_results.copy(), mars_results.copy()
        for d in (snaps, mars):
            d['key'] = d['protein'].str.strip().str.lower()
            d[num] = d[num].apply(pd.to_numeric, errors='coerce')
        snaps['condition'] = snaps['condition'].astype(str).str.strip()
        mars = mars.dropna(subset=num)

        # 1. check the name matching
        print("Only in one table:", set(snaps.key) ^ set(mars.key))
        print("SNAPS conditions:", sorted(snaps.condition.unique()))

        # 2. one row per protein (per condition for SNAPS); mean over runs within that group
        def summarise(df, tool, by):
            g = df.groupby(by).agg(protein=('protein', 'first'),
                                   n_assign=('assignments', 'max'),
                                   reliable=('reliable_assignments', 'mean'),
                                   errors=('errors', 'mean'),
                                   n_runs=('errors', 'size'))
            g['correct'] = g['reliable'] - g['errors']
            return g.add_suffix(f'_{tool}')

        s = summarise(snaps, 'snaps', ['key', 'condition']).reset_index(level='condition')
        r = summarise(mars, 'mars', ['key'])
        m = s.join(r, how='inner').reset_index()      # each SNAPS condition row paired with that protein's MARS row
        m['N'] = m[['n_assign_snaps', 'n_assign_mars']].max(axis=1)

        for t in ('snaps', 'mars'):
            m[f'correct_frac_{t}'] = m[f'correct_{t}'] / m['N']
            m[f'reliable_frac_{t}'] = m[f'reliable_{t}'] / m['N']
            m[f'err_rate_{t}'] = m[f'errors_{t}'] / m[f'reliable_{t}']
        m['diff'] = m['correct_frac_snaps'] - m['correct_frac_mars']   # >0 means SNAPS better

        # 3. per protein x condition: Fisher exact (approximate, see caveat)
        def fisher(r):
            N = int(r.N); a, b = round(r.correct_snaps), round(r.correct_mars)
            return stats.fisher_exact([[a, N - a], [b, N - b]])[1]
        m['p_fisher'] = m.apply(fisher, axis=1)

        pd.set_option('display.max_rows', None)
        print(m.sort_values(['condition', 'key'])[
            ['condition', 'protein_snaps', 'N', 'correct_frac_snaps', 'correct_frac_mars',
             'diff', 'err_rate_snaps', 'err_rate_mars', 'p_fisher']].round(3))

        # 4. overall, separately for each SNAPS condition vs MARS
        rng = np.random.default_rng(0)
        rows = []
        for cond, g in m.groupby('condition'):
            d = g['diff'].to_numpy()
            boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            try:
                p_w = stats.wilcoxon(d).pvalue
            except ValueError:                      # all differences zero
                p_w = np.nan
            p_t = stats.ttest_rel(g.correct_frac_snaps, g.correct_frac_mars).pvalue
            pooled_s = g.correct_snaps.sum() / g.N.sum()
            pooled_m = g.correct_mars.sum() / g.N.sum()
            rows.append(dict(condition=cond, n_proteins=len(d), mean_diff=d.mean(),
                             median_diff=np.median(d), ci_lo=lo, ci_hi=hi,
                             p_wilcoxon=p_w, p_ttest=p_t,
                             pooled_snaps=pooled_s, pooled_mars=pooled_m))
        summary = pd.DataFrame(rows).set_index('condition')
        print(summary.round(3))
        return m


    m = _()
    return (m,)


@app.cell
def _(m):
    import matplotlib.pyplot as plt
    import numpy as np

    c_snaps, c_mars = '#1f77b4', '#d95f02'
    rng = np.random.default_rng(0)

    def plot_condition(cond, xlim_diff=None):
        plot = m[m['condition'] == cond].sort_values('diff').reset_index(drop=True)
        y = np.arange(len(plot))
        labels = plot['protein_snaps']

        fig, axes = plt.subplots(1, 3, figsize=(18, max(5, 0.45 * len(plot) + 2)),
                                 gridspec_kw={'width_ratios': [1.4, 1, 1.1]})
        fig.suptitle(f'SNAPS condition: {cond}  vs  MARS', fontsize=14, y=1.02)

        # --- A. dumbbell: correct fraction per protein ---
        ax = axes[0]
        ax.hlines(y, plot.correct_frac_mars, plot.correct_frac_snaps, color='grey', alpha=.5, zorder=1)
        ax.scatter(plot.correct_frac_snaps, y, color=c_snaps, s=60, label='SNAPS', zorder=2)
        ax.scatter(plot.correct_frac_mars, y, color=c_mars, s=60, label='MARS', zorder=2)
        for yi, p in zip(y, plot.p_fisher):
            star = '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else ''
            if star:
                ax.text(1.02, yi, star, va='center', ha='left', fontsize=10)
        ax.set_yticks(y); ax.set_yticklabels(labels)
        ax.set_xlim(0, 1.08); ax.set_xlabel('Correct assignments / total residues')
        ax.set_title('A. Correct fraction per protein\n(* = Fisher p<0.05, approximate)')
        ax.legend(loc='lower right'); ax.grid(axis='x', alpha=.3)

        # --- B. difference per protein with overall mean and bootstrap CI ---
        ax = axes[1]
        colors = [c_snaps if v > 0 else c_mars for v in plot['diff']]
        ax.barh(y, plot['diff'], color=colors, alpha=.85)
        ax.axvline(0, color='k', lw=.8)
        d = plot['diff'].to_numpy()
        boot = rng.choice(d, size=(10000, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax.axvspan(lo, hi, color='grey', alpha=.2,
                   label=f'Bootstrap 95% CI of mean\n({lo:.2f} to {hi:.2f})')
        ax.axvline(d.mean(), color='k', ls='--', label=f'Mean diff = {d.mean():.2f}')
        if xlim_diff is not None:
            ax.set_xlim(*xlim_diff)
        ax.set_yticks(y); ax.set_yticklabels([])
        ax.set_xlabel('SNAPS − MARS (correct fraction)')
        ax.set_title('B. Difference (>0: SNAPS better)')
        ax.legend(loc='lower right', fontsize=8); ax.grid(axis='x', alpha=.3)

        # --- C. completeness vs error-rate trade-off ---
        ax = axes[2]
        for t, col, lab in (('snaps', c_snaps, 'SNAPS'), ('mars', c_mars, 'MARS')):
            ax.scatter(plot[f'reliable_frac_{t}'], plot[f'err_rate_{t}'],
                       color=col, s=60, alpha=.85, label=lab)
        for _, row in plot.iterrows():
            ax.plot([row.reliable_frac_snaps, row.reliable_frac_mars],
                    [row.err_rate_snaps, row.err_rate_mars], color='grey', alpha=.3, zorder=0)
        ax.set_xlabel('Reliable assignments / total residues')
        ax.set_ylabel('Errors / reliable assignments')
        ax.set_title('C. Completeness vs error rate\n(lower right is best)')
        ax.legend(); ax.grid(alpha=.3)

        plt.tight_layout()
        plt.show()

    # same x-range on panel B in every figure so the conditions are directly comparable
    lim = np.abs(m['diff']).max() * 1.1
    for cond in sorted(m['condition'].unique()):
        plot_condition(cond, xlim_diff=(-lim, lim))
    return


if __name__ == "__main__":
    app.run()
