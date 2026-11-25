"""
https://huggingface.co/datasets/hws-lab/rBMRB
"""

from datasets import load_dataset


def get_rbmrb():
    dataset = load_dataset('hws-lab/rBMRB', 'PANAV', split='train')
    return list(
        dict.fromkeys(
            entry_id.rsplit('_', 1)[0] for entry_id in dataset['entry_ID']
        )
    )
