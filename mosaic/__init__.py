import os
from typing import Union, List, Dict
from torch.utils.data import ConcatDataset
from .models import from_pretrained
from .constants import num_subjects, subject_id_to_file_mapping, BASE_URL, prefix, default_betas_folder
from .stiminfo import get_stiminfo
from .datasets import SingleSubjectDataset
from .utils.download import download_file, check_if_need_to_download
from .utils.folder import make_folder_if_does_not_exist
from .datasets.single_subject import validate_dataset_name, validate_subject_id

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
    


def download(
    names_and_subjects: Dict[str, Union[List[int], str]],
    folder: str = "./mosaic_dataset",
) -> List[str]:
    filenames = []

    # Validate all dataset names first
    for dataset_name in names_and_subjects.keys():
        validate_dataset_name(dataset_name)

    print(f"Downloading datasets to: {os.path.abspath(folder)}\n")
    make_folder_if_does_not_exist(folder=folder)

    for dataset_name, subject_ids_input in names_and_subjects.items():
        # Resolve subject IDs
        if subject_ids_input == "all":
            subject_ids = list(subject_id_to_file_mapping[dataset_name].keys())
        else:
            assert isinstance(subject_ids_input, list), (
                f"For dataset '{dataset_name}', subject_ids must be a list of ints or 'all', "
                f"got {type(subject_ids_input)}"
            )
            subject_ids = subject_ids_input

        # Validate each subject ID
        for subject_id in subject_ids:
            validate_subject_id(dataset_name=dataset_name, subject_id=subject_id)


        for subject_id in subject_ids:
            filename = subject_id_to_file_mapping[dataset_name][subject_id]
            save_path = os.path.join(folder, filename)
            filenames.append(save_path)

            if check_if_need_to_download(filename=save_path):
                remote_path = os.path.join(default_betas_folder, dataset_name, filename)
                print(f" Downloading {dataset_name} - Subject {subject_id}")
                download_file(
                    base_url=BASE_URL,
                    file=remote_path,
                    save_as=save_path,
                )
                print(f"{prefix} Saved to {save_path}")
            else:
                print(f"{prefix} File exists: {save_path}")

    print("All downloads completed.")
    return filenames