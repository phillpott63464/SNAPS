
import re

import pandas as pd

from modules.bmrb import run_mars_set
from modules.results import read_reference_assignment

# BMRB_NAMES = {
#     5471: "Malate synthase G",
# } example
def get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR):
    # try:
    mars_data_files = run_mars_set(BMRB_IDS, DATA_DIR)
    mars_data = format_mars(mars_data_files)

    # except OSError:
    #     print('using default')
    #     mars_data = {
    #         "I": {
    #             "Malate synthase G | C′, Cα, Cβ": (652, 639, 0),
    #             "Malate synthase G | Cα, Cβ*": (500, 207, 0),
    #             "Maltose binding protein | Cα, Cβ": (323, 303, 0),
    #             "Rous Sarcoma Virus capsid | C′, Cα, Cβ": (214, 205, 0),
    #             "Human carbonic anhydrase I | C′, Cα, Cβ": (242, 235, 0),
    #             "N-terminal domain of enzyme I (EIN) | Cα, Cβ": (246, 232, 0),
    #             "E-cadherin domains II and III | Cα, Cβ": (116, 77, 1),
    #             "Human prion protein | Cα, Cβ": (138, 103, 0),
    #             "Superoxide dismutase | C′, Cα, Cβ": (112, 101, 0),
    #             "Superoxide dismutase | Cα, Cβ": (111, 101, 0),
    #             "Calmodulin/M13 | Cα, C′": (97, 37, 0),
    #             "Profilin | Cα, Cβ": (130, 132, 2),
    #             "E. coli EmrE | C′, Cα, Cβ": (61, 35, 0),
    #             "Human ubiquitin | Cα, Cβ": (72, 72, 0),
    #             "Human ubiquitin | Cα only": (32, 9, 0),
    #             "Z domain | Cα, Cβ, Hα†": (65, 65, 0),
    #             "Z domain | Cα, Cβ‡": (57, 34, 0),
    #             "Tir110 | C′, Cα, Cβ§": (91, 80, 0),
    #         },
    #         "II": {
    #             "Malate synthase G | C′, Cα, Cβ": (652, 639, 0),
    #             "Malate synthase G | Cα, Cβ*": (639, 584, 2),
    #             "Maltose binding protein | Cα, Cβ": (333, 324, 0),
    #             "Rous Sarcoma Virus capsid | C′, Cα, Cβ": (218, 207, 0),
    #             "Human carbonic anhydrase I | C′, Cα, Cβ": (242, 237, 0),
    #             "N-terminal domain of enzyme I (EIN) | Cα, Cβ": (246, 246, 0),
    #             "E-cadherin domains II and III | Cα, Cβ": (134, 102, 0),
    #             "Human prion protein | Cα, Cβ": (155, 127, 0),
    #             "Superoxide dismutase | C′, Cα, Cβ": (112, 104, 0),
    #             "Superoxide dismutase | Cα, Cβ": (112, 104, 0),
    #             "Calmodulin/M13 | Cα, C′": (144, 142, 0),
    #             "Profilin | Cα, Cβ": (132, 132, 0),
    #             "E. coli EmrE | C′, Cα, Cβ": (70, 58, 0),
    #             "Human ubiquitin | Cα, Cβ": (72, 72, 0),
    #             "Human ubiquitin | Cα only": (58, 18, 0),
    #             "Z domain | Cα, Cβ, Hα†": None,
    #             "Z domain | Cα, Cβ‡": None,
    #             "Tir110 | C′, Cα, Cβ§": None,
    #         },
    #         "III": {
    #             "Malate synthase G | C′, Cα, Cβ": (651, 623, 0),
    #             "Malate synthase G | Cα, Cβ*": (622, 511, 0),
    #             "Maltose binding protein | Cα, Cβ": (330, 313, 1),
    #             "Rous Sarcoma Virus capsid | C′, Cα, Cβ": (218, 199, 0),
    #             "Human carbonic anhydrase I | C′, Cα, Cβ": (242, 225, 0),
    #             "N-terminal domain of enzyme I (EIN) | Cα, Cβ": (248, 245, 0),
    #             "E-cadherin domains II and III | Cα, Cβ": (136, 70, 1),
    #             "Human prion protein | Cα, Cβ": (154, 118, 0),
    #             "Superoxide dismutase | C′, Cα, Cβ": (111, 100, 0),
    #             "Superoxide dismutase | Cα, Cβ": (112, 103, 0),
    #             "Calmodulin/M13 | Cα, C′": (136, 119, 0),
    #             "Profilin | Cα, Cβ": (132, 123, 0),
    #             "E. coli EmrE | C′, Cα, Cβ": (64, 50, 0),
    #             "Human ubiquitin | Cα, Cβ": (72, 70, 0),
    #             "Human ubiquitin | Cα only": (58, 9, 0),
    #             "Z domain | Cα, Cβ, Hα†": None,
    #             "Z domain | Cα, Cβ‡": None,
    #             "Tir110 | C′, Cα, Cβ§": None,
    #         },
    #     }
    # except: # any other error
    #     print('Mars data set creation failed for a reason other than non prescence of podman')

    rows = []
    for condition, results_by_bmrb in mars_data.items():
        for bmrb_id, (assignments, reliable, errors) in results_by_bmrb.items():
            rows.append(
                {
                    "protein": lookup_name(bmrb_id, BMRB_NAMES),
                    'bmrb_id': str(bmrb_id),
                    "chemical_shifts_used": 'all',  # Not changing in this output
                    "condition": str(condition),
                    "assignments": assignments,
                    "reliable_assignments": reliable,
                    "errors": errors,
                }
            )

    print(mars_data_files)

    mars_results = pd.DataFrame(rows)
    mars_results[["assignments", "reliable_assignments", "errors"]] = mars_results[
        ["assignments", "reliable_assignments", "errors"]
    ].astype("Int64")

    return mars_results


def format_mars(mars_files: dict):
    output = {}

    for condition, proteins in mars_files.items():
        output[condition] = {}

        for bmrb_id, files in proteins.items():
            reference = read_reference_assignment(files["nef_file"])
            output[condition][bmrb_id] = parse_assignment_aa(
                files["assignment_AA"],
                reference,
            )

    return output


def parse_assignment_aa(path, reference):
    reference_residues = dict(
        zip(
            reference["sequence_code"].astype(str),
            reference["residue_name"].str.upper(),
        )
    )

    assignments = 0
    reliable = 0
    errors = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 2:
                continue

            target = re.fullmatch(r"([A-Z]{3})_(\d+)", fields[0])
            assigned = re.fullmatch(r"[A-Z]+_(\d+)_([A-Z]{3})", fields[1])
            if not target or not assigned:
                continue

            target_name, target_code = target.groups()
            assigned_code, assigned_name = assigned.groups()

            assignments += 1
            if len(fields) >= 3 and fields[2] == "(H)":
                reliable += 1

            # Only score assignments that can be checked against the NEF.
            reference_name = reference_residues.get(assigned_code)
            if reference_name is None:
                continue

            if (
                assigned_code != target_code
                or assigned_name != target_name
                or assigned_name != reference_name
            ):
                errors += 1

    return assignments, reliable, errors

def lookup_name(bmrb_id, BMRB_NAMES):
    # Handle IDs stored as either integers or strings in BMRB_NAMES.
    name = BMRB_NAMES.get(bmrb_id)
    if name is None:
        name = BMRB_NAMES.get(str(bmrb_id))
    if name is None:
        try:
            name = BMRB_NAMES.get(int(bmrb_id))
        except (TypeError, ValueError):
            pass
    if name is None:
        raise KeyError(f"No protein name found for BMRB ID {bmrb_id!r}")
    return name
