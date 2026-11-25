"""
https://huggingface.co/datasets/hws-lab/rBMRB
"""

from pathlib import Path
from typing import Optional

from datasets import load_dataset

from modules.bmrb import process_bmrb, with_psipred_dir


@with_psipred_dir
def get_rbmrb(
    data_dir: Optional[Path] = None,
    number: Optional[int] = None,
    split: str = 'train',
    psipred_dir: Optional[Path] = None,
):
    if split not in ['train', 'validation', 'test']:
        raise ValueError("Split doesn't exist")
    dataset = load_dataset('hws-lab/rBMRB', 'PANAV', split=split)
    keys = list(
        dict.fromkeys(
            entry_id.rsplit('_', 1)[0] for entry_id in dataset['entry_ID']
        )
    )

    if number is None:
        return keys

    if data_dir is None:
        raise ValueError('Requires data_dir')

    bmrb_ids = []
    while len(bmrb_ids) < number and len(keys) > 0:
        key = keys.pop(0)
        if process_bmrb(
            bmrb_id=key, data_dir=data_dir / key, psipred_dir=psipred_dir
        ):
            bmrb_ids.append(key)

    return bmrb_ids
