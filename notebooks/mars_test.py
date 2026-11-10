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
        4354: "Maltose-binding protein",
        4384: "Rous sarcoma virus capsid protein",
        4022: "Carbonic anhydrase I",
        4106: "Enzyme I N-terminal domain",
        4457: "E-cadherin",
        4402: "Human prion protein",
        4341: "Iron superoxide dismutase",
        547: "Calmodulin",
        4082: "Profilin",
        4136: "EmrE multidrug resistance protein",
    }
    DATA_DIR = Path(__file__).resolve().parent / "mars_test_set"
    OUTPUT_DIR = Path(__file__).resolve().parent / "mars_test_result"
    return BMRB_IDS, BMRB_NAMES, DATA_DIR, OUTPUT_DIR


@app.cell(hide_code=True)
def _(mars_results, mo):
    mo.md(rf"""
    # Download Test Data from BMRB for Mars Test Dataset

    ## Their Results:

    {mo.md(mars_results.to_markdown(index=False))}

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
        if all(
            (OUTPUT_DIR / f'{bmrb_id}-test-result-{strategy}.{file_type}').exists()
            for strategy in strategies for file_type in file_types
        ):
            continue

        process_bmrb(bmrb_id)

        base_overrides = {
            'obs_file': DATA_DIR / f'{bmrb_id}.nef',
            'pred_file': DATA_DIR / f'{bmrb_id}-shiftx2.nef:shiftx2',
            'sequence_file': DATA_DIR / f'{bmrb_id}.fasta',
        }

        for strategy in strategies:
            for file_type in file_types:
                if (OUTPUT_DIR / f'{bmrb_id}-test-result-{strategy}.{file_type}').exists():
                    continue

                config = build_config(
                    cli_overrides = {
                        'strategy': strategy,
                        **({'SS_num': True} if strategy == 'triplet' else {}),
                        **({'output_type': 'nef'} if file_type == 'nef' else {'output_type': 'snaps'}),
                        'output_file': OUTPUT_DIR / f'{bmrb_id}-test-result-{strategy}.{file_type}',
                        **base_overrides
                    }
                )

            run_pipeline(config)
    return


@app.cell(hide_code=True)
def _(mo, shift_results):
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

    {mo.md(shift_results.to_markdown(index=False))}
    """)
    return


@app.cell
def _(BMRB_IDS, BMRB_NAMES, OUTPUT_DIR, Path):
    # Create our results

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
    CONDITIONS = {
        "I": lambda bmrb_id: OUTPUT_DIR / f"{bmrb_id}-test-result-direct.txt",
        "II": lambda bmrb_id: OUTPUT_DIR / f"{bmrb_id}-test-result-iterative.txt",
        "III": lambda bmrb_id: OUTPUT_DIR / f"{bmrb_id}-test-result-triplet.txt",
    }


    def assignment_counts(result, chain="A", label_to_residue=None):
        assigned = result.loc[
            result["assigned"] 
            & result["chain_code"].eq(chain) 
            & result["residue_name"].ne("PRO") # Remove prolines because MARS doesn't predict them (I am fairly certain)
        ].copy()

    
        if "merit" in assigned.columns:
            reliable = assigned.loc[assigned["merit"].le(RELIABLE_MERIT)].copy()
        elif "Confidence" in assigned.columns:
            reliable = assigned.loc[
                assigned["Confidence"].isin(["High", "Medium"])
            ].copy()
        else:
            raise ValueError("Result has neither merit nor Confidence")

        # Numeric labels are assumed to be original residue numbers,
        # matching the assumption in your existing comparison.
        reference = reliable["spin_system"].copy()

        # Optional explicit mappings for labels such as DSS_40.
        if label_to_residue is not None:
            labels = (
                reliable["unassigned_sequence_code"]
                .astype(str)
                .str.strip()
            )
            mapped = pd.to_numeric(
                labels.map(label_to_residue), errors="coerce"
            )
            reference = mapped.combine_first(reference)

        predicted = reliable["residue_number"]

        # Do not report an error count unless every reliable assignment
        # can be evaluated against a reference residue.
        evaluable = (
            reference.notna()
            & predicted.notna()
            & reference.mod(1).eq(0)
            & predicted.mod(1).eq(0)
        )

        errors = (
            str(int(reference.ne(predicted).sum()))
            if evaluable.all()
            else "—"
        )

        return f"{len(assigned)} / {len(reliable)} / {errors}"


    def markdown_cell(value):
        return str(value).replace("|", r"\|").replace("\n", " ")


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
                bmrb_id, (f"{BMRB_NAMES[bmrb_id]}", "—")
            )
            row = {
                "Protein": protein,
                "Chemical shifts used": shifts_used,
            }

            for condition in ("I", "II", "III"):
                column = f"Condition {condition}: assignments / reliable / errors"
                path_factory = conditions.get(condition)

                if path_factory is None:
                    row[column] = "—"
                    continue

                path = Path(path_factory(bmrb_id))
                if not path.is_file():
                    row[column] = "—"
                    continue

                result = read_test_result(path)
                row[column] = assignment_counts(
                    result,
                    chain=chain,
                    label_to_residue=label_maps.get(bmrb_id),
                )

            rows.append(row)

        return pd.DataFrame(rows)

    shift_results = assignment_table(
        BMRB_IDS,
        CONDITIONS,
        PROTEIN_INFO,
    )
    return pd, shift_results


@app.cell
def _(pd, shift_results):
    # Published comparison table, using the same columns and row format as shift_results.
    mars_results = pd.DataFrame(
        [
            ("Malate synthase G", "C′, Cα, Cβ", "652 / 639 / 0", "652 / 639 / 0", "651 / 623 / 0"),
            ("Malate synthase G", "Cα, Cβ*", "500 / 207 / 0", "639 / 584 / 2", "622 / 511 / 0"),
            ("Maltose binding protein", "Cα, Cβ", "323 / 303 / 0", "333 / 324 / 0", "330 / 313 / 1"),
            ("Rous Sarcoma Virus capsid", "C′, Cα, Cβ", "214 / 205 / 0", "218 / 207 / 0", "218 / 199 / 0"),
            ("Human carbonic anhydrase I", "C′, Cα, Cβ", "242 / 235 / 0", "242 / 237 / 0", "242 / 225 / 0"),
            ("N-terminal domain of enzyme I (EIN)", "Cα, Cβ", "246 / 232 / 0", "246 / 246 / 0", "248 / 245 / 0"),
            ("E-cadherin domains II and III", "Cα, Cβ", "116 / 77 / 1", "134 / 102 / 0", "136 / 70 / 1"),
            ("Human prion protein", "Cα, Cβ", "138 / 103 / 0", "155 / 127 / 0", "154 / 118 / 0"),
            ("Superoxide dismutase", "C′, Cα, Cβ", "112 / 101 / 0", "112 / 104 / 0", "111 / 100 / 0"),
            ("Superoxide dismutase", "Cα, Cβ", "111 / 101 / 0", "112 / 104 / 0", "112 / 103 / 0"),
            ("Calmodulin/M13", "Cα, C′", "97 / 37 / 0", "144 / 142 / 0", "136 / 119 / 0"),
            ("Profilin", "Cα, Cβ", "130 / 132 / 2", "132 / 132 / 0", "132 / 123 / 0"),
            ("E. coli EmrE", "C′, Cα, Cβ", "61 / 35 / 0", "70 / 58 / 0", "64 / 50 / 0"),
            ("Human ubiquitin", "Cα, Cβ", "72 / 72 / 0", "72 / 72 / 0", "72 / 70 / 0"),
            ("Human ubiquitin", "Cα only", "32 / 9 / 0", "58 / 18 / 0", "58 / 9 / 0"),
            ("Z domain", "Cα, Cβ, Hα†", "65 / 65 / 0", "—", "—"),
            ("Z domain", "Cα, Cβ‡", "57 / 34 / 0", "—", "—"),
            ("Tir110", "C′, Cα, Cβ§", "91 / 80 / 0", "—", "—"),
        ],
        columns=shift_results.columns,
    )
    return (mars_results,)


@app.cell
def _():
    # Compare the tables

    # Conditions I, II, and III in the shift_results is just strategies
    return


@app.cell(hide_code=True)
def _(mars_results, mo, shift_results):
    mo.md(rf"""
    Very simple comparison for now

    # SNAPS

    {mo.md(shift_results.to_markdown(index=False))}

    # MARS

    {mo.md(mars_results.to_markdown(index=False))}
    """)
    return


if __name__ == "__main__":
    app.run()
