import pytest
import mosaic
from mosaic.constants import subject_id_to_file_mapping
from mosaic.datasets.single_subject import SingleSubjectDataset
from mosaic.utils.json import load_json
from torch.utils.data import ConcatDataset

DATASET_FOLDER = load_json("tests/testing_config.json")["dataset_folder"]

all_dataset_names = list(subject_id_to_file_mapping.keys())

@pytest.mark.parametrize("dataset_name", all_dataset_names)
@pytest.mark.parametrize("parse_betas", [True, False])
def test_data(
    dataset_name: str,
    parse_betas: bool,
):
    num_subjects = len(subject_id_to_file_mapping[dataset_name])

    # Load each subject individually
    for subject_id in range(1, num_subjects + 1):
        dataset = mosaic.load(
            names_and_subjects={dataset_name: [subject_id]},
            parse_betas=parse_betas,
            folder=DATASET_FOLDER
        )
        assert isinstance(dataset, SingleSubjectDataset), f"Expected dataset to be instance of SingleSubjectDataset, but got {type(dataset)}"

    # Load all subjects together
    dataset = mosaic.load(
        names_and_subjects={dataset_name: list(range(1, num_subjects + 1))},
        parse_betas=parse_betas,
        folder=DATASET_FOLDER
    )
    assert isinstance(dataset, ConcatDataset), f"Expected dataset to be instance of ConcatDataset, but got {type(dataset)}"
    
    # check if the "all" functionality is working
    dataset_all = mosaic.load(
        names_and_subjects={dataset_name: "all"},
        parse_betas=parse_betas,
        folder=DATASET_FOLDER
    )
    assert isinstance(dataset_all, ConcatDataset), f"Expected dataset to be instance of ConcatDataset, but got {type(dataset_all)}"

    assert len(dataset) == len(dataset_all), f"Expected dataset length {len(dataset)} to equal dataset_all length {len(dataset_all)}"
