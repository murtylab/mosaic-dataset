import requests
import os
import pandas as pd
from .constants import BASE_URL, stiminfo_folder
from .utils.download import download_file

file_mapping = {
    "BOLD5000": "b5000_stiminfo.tsv",
    "deeprecon": "deeprecon_stiminfo.tsv",
    "GOD": "god_stiminfo.tsv",
    "NSD": "nsd_stiminfo.tsv",
    "THINGS": "things_stiminfo.tsv",
    "BMD": "bmd_stiminfo.tsv",
    "NOD": "nod_stiminfo.tsv",
    "HAD": "had_stiminfo.tsv",
}

def get_stiminfo(dataset_name: str, folder: str = "./mosaic_stiminfo") -> str:

    assert dataset_name in file_mapping, f"Invalid dataset_name {dataset_name}. Must be one of {list(file_mapping.keys())}"
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    stiminfo_filename = os.path.join(folder, file_mapping[dataset_name])

    if not os.path.exists(stiminfo_filename):
        file_in_s3 =  os.path.join(stiminfo_folder, file_mapping[dataset_name])
        url = BASE_URL + "/" + file_in_s3
        response = requests.head(url)
        assert response.status_code == 200, f"URL {url} is not valid or reachable."
        download_file(
            base_url=BASE_URL,
            file=file_in_s3,
            save_as=stiminfo_filename,
        )
    else:
        print(f"Stiminfo file already exists at {stiminfo_filename}, skipping download.")
    
    ## stiminfo is a tsv file, load it using pandas as return a DataFrame
    
    stiminfo = pd.read_csv(stiminfo_filename, sep="\t")
    return stiminfo