import os
from typing import Union, List

from .datasets import (
    BOLD5000SingleSubject,
    GenericObjectDecodingSingleSubject,
    NSDSingleSubject,
    ThingsFMRISingleSubject,
    DeepReconSingleSubject,
)

name_to_dataset_mapping = {
    "bold5000": BOLD5000SingleSubject,
    "generic_object_decoding": GenericObjectDecodingSingleSubject,
    "nsd": NSDSingleSubject,
    "things_fmri": ThingsFMRISingleSubject,
    "deep_recon": DeepReconSingleSubject,
}

def load(
    name: str,
    subject_id: Union[List[int], str] = 1,
    folder: str = "data",
):
    ## assert name is valid
    assert name in name_to_dataset_mapping.keys(), \
        f"Dataset name {name} is not valid. Please choose from {list(name_to_dataset_mapping.keys())}."
    
    if isinstance(subject_id, str):
        assert subject_id == "all", \
            "If subject_id is a string, it must be 'all'."
        raise NotImplementedError("Loading all subjects is not implemented yet.")

    else:
        dataset_class = name_to_dataset_mapping[name]
        folder = os.path.join(
            folder, name
        )
        return dataset_class(
            folder=folder,
            subject_id=subject_id,
        )