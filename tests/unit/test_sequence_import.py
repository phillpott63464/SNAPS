import pytest
from refactor.SNAPSconfig import build_config
from refactor.SNAPSimporter import SNAPSImporter
from pathlib import Path
import pandas as pd

TEST_DATA = Path(__file__).resolve().parents[1] / 'test_data'

@pytest.fixture
def sequence_file(tmp_path):
   data = """
GPAM TDYIQLLSEI AKEQGFNITY LDIDELSANG QYQCRAELST SPITVCHGSG ISCGNAQSDA AHNALQYLKI IAERK
   """
   with open(tmp_path / 'fasta.fasta', 'w') as f:
       f.write(data)

   return tmp_path / 'fasta.fasta'

def test_sequence_import(sequence_file):
    cli_overrides = {
        'obs_file': TEST_DATA / 'naps_shifts.txt',
        'pred_file': TEST_DATA / 'P3a_L273R_sparta.tab',
        'sequence_file': sequence_file,
    }
    config = build_config(cli_overrides=cli_overrides)

    importer = SNAPSImporter(config=config)
    data = importer.import_data(
        obs_file=config.obs_file,
        pred_file=config.pred_file,
        obs_format=config.obs_format,
        pred_format=config.pred_format,
        obs_frame=config.obs_frame,
        pred_frame=config.pred_frame,
        obs_chain=config.obs_chain,
        pred_chain=config.pred_chain,
        aa_restraints_file=config.aa_restraints_file,
        aa_restraints_frame=config.aa_restraints_frame,
    )

    expected_sequence = (
        "GPAM TDYIQLLSEI AKEQGFNITY LDIDELSANG "
        "QYQCRAELST SPITVCHGSG ISCGNAQSDA "
        "AHNALQYLKI IAERK"
    ).replace(" ", "")

    assert "".join(data.sequence["Res_type"]) == expected_sequence

    assert len(data.sequence) == len(expected_sequence)

    assert data.sequence["Seq_N"].tolist() == list(
        range(1, len(expected_sequence) + 1)
    )

    assert data.sequence["Res_N"].tolist() == list(
        range(1, len(expected_sequence) + 1)
    )

    assert data.sequence["Res_type"].tolist() == list(expected_sequence)

    assert data.sequence["Res_name_m1"].iloc[1] == (
        data.sequence["Res_name"].iloc[0]
    )

    assert data.sequence["Res_name_p1"].iloc[0] == (
        data.sequence["Res_name"].iloc[1]
    )

    assert pd.isna(data.sequence["Res_name_m1"].iloc[0])
    assert pd.isna(data.sequence["Res_name_p1"].iloc[-1])