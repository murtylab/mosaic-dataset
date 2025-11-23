import os
from typing import Union, List
from torch.utils.data import ConcatDataset
from .models import from_pretrained
from .constants import num_subjects, subject_id_to_file_mapping
from .stiminfo import get_stiminfo
from .datasets import SingleSubjectDataset


def load_single_dataset(
    name: str,
    subject_id: int = 1,
    folder: str = "./mosaic_dataset",
    parse_betas: bool = True,
):
    dataset = SingleSubjectDataset(
        folder=folder,
        dataset_name=name,
        subject_id=subject_id,
        parse_betas=parse_betas,
    )
    return dataset


def load(
    names_and_subjects: dict[str, Union[List[int], str]],
    folder: str = "./mosaic_dataset",
    parse_betas: bool = True,
):
    all_datasets = []

    for dataset_name, subject_ids in names_and_subjects.items():
        if subject_ids == "all":
            subject_ids = list(range(1, len(subject_id_to_file_mapping[dataset_name]) + 1))
        else:
            assert isinstance(subject_ids, list), f"subject_ids must be a list or 'all', got {type(subject_ids)}"

        for subject_id in subject_ids:
            dataset = load_single_dataset(
                name=dataset_name,
                subject_id=subject_id,
                folder=folder,
                parse_betas=parse_betas,
            )
            all_datasets.append(dataset)

    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        combined_dataset = ConcatDataset(all_datasets)
        return combined_dataset
    
    