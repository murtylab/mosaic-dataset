import os
import requests
from tqdm import tqdm


def download_file(base_url: str, file: str, save_as: str):
    url = f"{base_url}/{file}"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(save_as, "wb") as f, tqdm(
                desc=f"\033[92m[MOSAIC]\033[0m Downloading {file}",
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
