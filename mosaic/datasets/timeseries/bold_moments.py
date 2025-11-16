import os
import requests
from tqdm import tqdm
import xml.etree.ElementTree as ET
from ...constants import task_folder, BASE_URL
from ...utils.download import download_file


def list_s3_folder(base_url: str, prefix: str):
    """
    List all objects under a public S3 prefix by parsing its XML index.
    Works for S3 static website endpoints like mosaicfmri.s3.amazonaws.com.
    """
    # S3 bucket listing URL
    list_url = f"{base_url}?prefix={prefix}"

    resp = requests.get(list_url)
    resp.raise_for_status()

    # Parse the XML list
    root = ET.fromstring(resp.text)
    namespace = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

    keys = []
    for contents in root.findall("s3:Contents", namespace):
        key = contents.find("s3:Key", namespace).text
        # Skip directory placeholders
        if not key.endswith("/"):
            keys.append(key)

    return keys


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
