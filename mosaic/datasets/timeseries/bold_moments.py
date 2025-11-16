import os
import requests
from tqdm import tqdm
from ...constants import task_folder, BASE_URL
from ...utils.download import download_file
from ...utils.aws import list_s3_folder

def download_bold_moments_timeseries_data(folder: str):
    """
    Downloads ALL files in the BOLD5000 temporal filtering folder,
    using the same API style you provided.
    """

    # Build S3 prefix folder
    folder_in_s3 = os.path.join(
        task_folder,
        "BOLDMomentsDataset"
    )

    # ---- 1. List files under the S3 prefix ----
    all_files = list_s3_folder(BASE_URL, folder_in_s3)

    print(f"Found {len(all_files)} files to download.")

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    downloaded_paths = []

    for file_in_s3 in tqdm(all_files, desc="Downloading BOLD Moments timeseries data"):

        filename = os.path.basename(file_in_s3)
        url = os.path.join(BASE_URL, file_in_s3)

        # ---- 2. Validate file exists ----
        response = requests.head(url)
        assert response.status_code == 200, f"URL {url} is not valid or reachable."

        # ---- 3. Download file ----
        local_path = os.path.join(folder, filename)
        download_file(
            base_url=BASE_URL,
            file=file_in_s3,
            save_as=local_path,
        )

        downloaded_paths.append(local_path)

    return downloaded_paths
