
import requests
import os
import pandas as pd
from .constants import BASE_URL, participantinfo_folder
from .utils.download import download_file

## small inconsistency here: THINGS_fmri here refers to THINGS everywhere else in the codebase.
## maybe we should fix this down the line
valid_datasets = {"BOLD5000", "BOLDMomentsDataset", "deeprecon", "GenericObjectDecoding", "HumanActionsDataset", "NaturalObjectDataset", "NaturalScenesDataset", "THINGS_fmri"}

def get_participantinfo(dataset_name: str, folder: str = "./mosaic_participantinfo") -> str:
    
    if dataset_name == 'shared':
        participantinfo_filename = os.path.join(folder, 'participants_shared.tsv')
        file_in_s3 = os.path.join(participantinfo_folder, 'participants_shared.tsv')
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
    else:
        assert dataset_name in valid_datasets, f"Invalid dataset_name {dataset_name}. Must be 'shared' to get the overlapping subjects across datasets, or one of {list(valid_datasets)} for a specific dataset."
        participantinfo_filename = os.path.join(folder, dataset_name, 'participants.tsv')
        file_in_s3 = os.path.join(participantinfo_folder, dataset_name, 'participants.tsv')
        if not os.path.exists(os.path.join(folder, dataset_name)):
            os.makedirs(os.path.join(folder, dataset_name), exist_ok=True)

    if not os.path.exists(participantinfo_filename):
        url = BASE_URL + "/" + file_in_s3
        response = requests.head(url)
        assert response.status_code == 200, f"URL {url} is not valid or reachable."
        download_file(
            base_url=BASE_URL,
            file=file_in_s3,
            save_as=participantinfo_filename,
        )
    else:
        print(f"Participantinfo file already exists at {participantinfo_filename}, skipping download.")
    
    ## stiminfo is a tsv file, load it using pandas as return a DataFrame
    
    participantinfo = pd.read_csv(participantinfo_filename, sep="\t")
    return participantinfo