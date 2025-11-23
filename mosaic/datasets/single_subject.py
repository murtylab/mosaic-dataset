import os
import h5py
import numpy as np
from ..utils.download import download_file, check_if_need_to_download
from ..utils.folder import make_folder_if_does_not_exist
from ..constants import BASE_URL, default_betas_folder, subject_id_to_file_mapping
from ..utils.parcellation import parse_betas

def validate_dataset_name(dataset_name: str):
    valid_dataset_names = list(
        subject_id_to_file_mapping.keys()
    )
    assert dataset_name in valid_dataset_names, (
        f"Dataset name {dataset_name} is not valid. "
        f"Please choose from {valid_dataset_names}."
    )

def validate_subject_id(dataset_name: str, subject_id: int):
    valid_subject_ids = list(
        subject_id_to_file_mapping[dataset_name].keys()
    )
    assert subject_id in valid_subject_ids, (
        f"Subject ID {subject_id} is not valid for dataset {dataset_name}. "
        f"Please choose from {valid_subject_ids}."
    )


class SingleSubjectDataset:
    def __init__(
        self,
        folder: str,
        dataset_name: str,
        subject_id: int = 1,
        parse_betas: bool = True,
    ):

        validate_dataset_name(dataset_name=dataset_name)
        validate_subject_id(
            dataset_name=dataset_name,
            subject_id=subject_id,
        )

        self.parse_betas = parse_betas
        self.dataset_name = dataset_name
        self.folder = folder
        self.subject_id = subject_id
        self.filename = os.path.join(
            self.folder, subject_id_to_file_mapping[dataset_name][self.subject_id]
        )
        need_to_download = check_if_need_to_download(filename=self.filename)

        if need_to_download:
            make_folder_if_does_not_exist(folder=self.folder)
            file = os.path.join(
                default_betas_folder,
                dataset_name,
                subject_id_to_file_mapping[dataset_name][self.subject_id],
            )
            download_file(
                base_url=BASE_URL,
                file=file,
                save_as=self.filename,
            )
        else:
            print(f"Dataset {self.filename} already downloaded.")

        assert os.path.exists(
            self.filename
        ), f"File {self.filename} does not exist. Please check the download."
        ## the filename is an hdf5 file
        self.data = h5py.File(self.filename, "r")
        self.all_names = list(self.data["betas"].keys())

    def __getitem__(self, index: int) -> dict:

        item = np.array(self.data["betas"][self.all_names[index]])
        if self.parse_betas:
            item = parse_betas(betas=item)
        else:
            pass

        return {
            "name": self.all_names[index],
            "betas": item,
        }

    def __len__(self) -> int:
        return len(self.all_names)
