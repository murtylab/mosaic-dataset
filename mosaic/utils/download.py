import os
import requests
from tqdm import tqdm
import h5py
from ..constants import prefix

def check_if_need_to_download(filename: str):
    file_exists = os.path.exists(filename)
    need_to_download = False
    if file_exists:
        try:
            data = h5py.File(filename, "r")
        except OSError:
            need_to_download = True
    else:
        need_to_download = True

    return need_to_download

def check_if_url_exists(url: str) -> bool:
    try:
        response = requests.head(url)
        if response.status_code == 200:
            return True
        else:
            raise ValueError(f"URL ({url}) does not exist or returned status code {response.status_code}")
    except requests.RequestException as e:
        raise RuntimeError(f"Request failed: {e}")

def download_file(base_url: str, file: str, save_as: str):
    url = f"{base_url}/{file}"
    check_if_url_exists(url=url)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(save_as, "wb") as f, tqdm(
                desc=f"{prefix} Downloading {file}",
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"\033[92mSuccessfully downloaded\nFile:{file}\nTo:{save_as}\033[0m\n")
    except requests.RequestException as e:
        print(f"\033[91mFailed to download {file}: {e}\033[0m\n")
