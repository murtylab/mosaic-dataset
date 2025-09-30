import os
from typing import Union, List
from torch.utils.data import ConcatDataset
from .models import from_pretrained

from .datasets import (
    BOLD5000SingleSubject,
    GenericObjectDecodingSingleSubject,
    NSDSingleSubject,
    ThingsFMRISingleSubject,
    DeepReconSingleSubject,
    MultiSubjectDataset,
    BOLDMomentsSingleSubject,
    NODSingleSubject,
    HADSingleSubject,
)

name_to_dataset_mapping = {
    "bold5000": BOLD5000SingleSubject,
    "generic_object_decoding": GenericObjectDecodingSingleSubject,
    "nsd": NSDSingleSubject,
    "things_fmri": ThingsFMRISingleSubject,
    "deep_recon": DeepReconSingleSubject,
    "bold_moments": BOLDMomentsSingleSubject,
    "nod": NODSingleSubject,
    "had": HADSingleSubject,
}

num_subjects = {
    "bold5000": 4,
    "deeprecon": 3,
    "generic_object_decoding": 5,
    "nsd": 8,
    "things_fmri": 3,
    "deep_recon": 3,
    "bold_moments": 10,
    "nod": 30,
    "had": 30,
}


def load_single_dataset(
    name: str,
    subject_id: Union[List[int], str] = 1,
    folder: str = "./mosaic_dataset",
):
    ## assert name is valid
    assert (
        name in name_to_dataset_mapping.keys()
    ), f"Dataset name {name} is not valid. Please choose from {list(name_to_dataset_mapping.keys())}."

    if name != "all":
        if isinstance(subject_id, str):
            assert subject_id == "all", "If subject_id is a string, it must be 'all'."
            return MultiSubjectDataset(
                datasets=[
                    name_to_dataset_mapping[name](
                        folder=os.path.join(folder, name),
                        subject_id=subject_id,
                    )
                    for subject_id in range(1, num_subjects[name] + 1)
                ]
            )

        else:
            dataset_class = name_to_dataset_mapping[name]
            folder = os.path.join(folder, name)
            return dataset_class(
                folder=folder,
                subject_id=subject_id,
            )
    else:
        assert subject_id == "all", "If name is 'all', subject_id must be 'all'."
        return MultiSubjectDataset(
            datasets=[
                name_to_dataset_mapping[name](
                    folder=os.path.join(folder, name), subject_id=subject_id
                )
                for name in name_to_dataset_mapping.keys()
                for subject_id in range(1, num_subjects[name] + 1)
            ]
        )


def load(
    names_and_subjects: dict[str, Union[List[int], str]],
    folder: str = "./mosaic_dataset",
):
    assert isinstance(
        names_and_subjects, dict
    ), f"names_and_subjects must be a dict mapping dataset names (str) to subject_ids (list of integers or 'all'), but got: {type(names_and_subjects)}"

    for dataset_name in names_and_subjects.keys():
        assert (
            dataset_name in name_to_dataset_mapping.keys()
        ), f"Dataset name {dataset_name} is not valid. Please choose from {list(name_to_dataset_mapping.keys())}."
        subject_ids = names_and_subjects[dataset_name]
        assert isinstance(
            subject_ids, (list, str)
        ), f"For {dataset_name}, subject_ids must be a list of integers or 'all', but got: {type(subject_ids)}"
        if isinstance(subject_ids, list):
            for subject_id in subject_ids:
                assert isinstance(
                    subject_id, int
                ), f"For {dataset_name}, each subject_id must be an integer, but got: {type(subject_id)}"
                assert (
                    1 <= subject_id <= num_subjects[dataset_name]
                ), f"For {dataset_name}, subject_id {subject_id} is out of range for dataset {dataset_name}. Must be between 1 and {num_subjects[dataset_name]}."
        else:
            assert (
                subject_ids == "all"
            ), f"If subject_ids is a string, it must be 'all', but got: {subject_ids}"

    names = list(names_and_subjects.keys())
    subject_ids = list(names_and_subjects.values())

    all_datasets = []
    for name, subject_id in zip(names, subject_ids):
        if subject_id != "all":
            for id in subject_id:
                dataset = load_single_dataset(name=name, subject_id=id, folder=folder)
                all_datasets.append(dataset)
        else:
            assert (
                subject_id == "all"
            ), f"If subject_id is a string, it must be 'all', but got: {subject_id}"
            dataset = load_single_dataset(
                name=name, subject_id=subject_id, folder=folder
            )

        all_datasets.append(dataset)

    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return ConcatDataset(all_datasets)
