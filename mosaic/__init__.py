import os
from typing import Union, List

from .datasets import (
    BOLD5000SingleSubject,
    GenericObjectDecodingSingleSubject,
    NSDSingleSubject,
    ThingsFMRISingleSubject,
    DeepReconSingleSubject,
    MultiSubjectDataset
)

name_to_dataset_mapping = {
    "bold5000": BOLD5000SingleSubject,
    "generic_object_decoding": GenericObjectDecodingSingleSubject,
    "nsd": NSDSingleSubject,
    "things_fmri": ThingsFMRISingleSubject,
    "deep_recon": DeepReconSingleSubject,
}

num_subjects = {
    "bold5000": 2,
    "deeprecon": 3,
    "generic_object_decoding": 5,
    "nsd": 1,
    "things_fmri": 3
}

def load(
    name: str,
    subject_id: Union[List[int], str] = 1,
    folder: str = "data",
):
    ## assert name is valid
    assert name in name_to_dataset_mapping.keys(), \
        f"Dataset name {name} is not valid. Please choose from {list(name_to_dataset_mapping.keys())}."
    
    if name != "all":
        if isinstance(subject_id, str):
            assert subject_id == "all", \
                "If subject_id is a string, it must be 'all'."
            return MultiSubjectDataset(
                datasets=[
                    name_to_dataset_mapping[name](
                        folder=os.path.join(folder, name),
                        subject_id=subject_id,
                    )
                    for subject_id in range(1, num_subjects[name]+1)
                ]
            )

        else:
            dataset_class = name_to_dataset_mapping[name]
            folder = os.path.join(
                folder, name
            )
            return dataset_class(
                folder=folder,
                subject_id=subject_id,
            )
    else:
        assert subject_id == "all",\
            "If name is 'all', subject_id must be 'all'."
        return MultiSubjectDataset(
            datasets=[
                name_to_dataset_mapping[name](folder=os.path.join(folder, name), subject_id=subject_id)
                for name in name_to_dataset_mapping.keys()
                for subject_id in range(1, num_subjects[name]+1)
            ]
        )