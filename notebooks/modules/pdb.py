from pathlib import Path
import pandas as pd
from modules.results import read_test_result_snaps
from pymol import cmd
from Bio.PDB import PDBParser, PPBuilder
from Bio.SeqUtils import seq1

def colour_mars_set(bmrb_ids: list[int], data_dir: Path, output_dir: Path):
    results = []
    for bmrb_id in bmrb_ids:
        for strategy in ['direct']:
            results += [colour_by_confidence(data_dir / f'{bmrb_id}.pdb', output_dir / f'{bmrb_id}.pse', output_dir / f'{bmrb_id}-test-result-{strategy}.txt')]
    return results
    ...

def colour_by_confidence(pdb_path: Path, pse_path: Path, result_path: Path):
    result = read_test_result_snaps(result_path)
    result = result[["Res_N", "Confidence", "Res_type"]]

    pdb = PDBParser(QUIET=True).get_structure("s", pdb_path)
    chain = pdb[0]["A"]  # change chain ID

    # Build a sequence from the chain's amino-acid residues
    pdb_residues = [
        r for r in chain
        if r.id[0] == " " and "CA" in r
    ]
    pdb_seq = "".join(seq1(r.resname, custom_map={"UNK": "X"}) for r in pdb_residues)

    df_seq = "".join(
        result.loc[result["Res_type"].notna(), "Res_type"].astype(str)
    )

    print("PDB:", len(pdb_seq), pdb_seq)
    print("DF: ", len(df_seq), df_seq)

    raise AssertionError('BReak here')
    # Get dict of residues by confidence
    # load pdb
    # Colour residues by confidence (dark green for high, lighter green for medium, yellow for low, grey for undefined, red for unreliable (same colours as from SNAPSoutput)
    # save as pse