import os
import requests
from tqdm import tqdm

def download_file(base_url: str, file: str, save_as: str):
    url = f"{base_url}/{file}"
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))
            with open(save_as, 'wb') as f, tqdm(
                desc=file,
                total=total,
                unit='B',
                unit_scale=True,
                unit_divisor=1024
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    bar.update(len(chunk))
        print(f"\033[92mSuccessfully downloaded {file}\033[0m\n")
    except requests.RequestException as e:
        print(f"\033[91mFailed to download {file}: {e}\033[0m\n")



def download_files_from_cloudfront():
    
    FILES = [
        
        "sub-01_GOD_crc32-3e714d56.hdf5",
        "sub-02_GOD_crc32-a9f461c6.hdf5",
        "sub-03_GOD_crc32-d5783090.hdf5",
        "sub-04_GOD_crc32-3723b278.hdf5",
        "sub-05_GOD_crc32-9761c38d.hdf5",
        "sub-01_NSD_crc32-32353cf9.hdf5",
        "sub-01_THINGS_crc32-a8b1d77b.hdf5",
        "sub-02_THINGS_crc32-3bc9008c.hdf5",
        "sub-03_THINGS_crc32-0ceefdd6.hdf5",
        "sub-01_deeprecon_crc32-f4d0e132.hdf5",
        "sub-02_deeprecon_crc32-137049f9.hdf5",
        "sub-03_deeprecon_crc32-c88ea8b8.hdf5"
    ]

    print(f"Downloading {len(FILES)} files...\n")

    for file in FILES:
        url = f"{CLOUDFRONT_URL}/{file}"
        local_path = os.path.join(".", file)
        print(f"Downloading {file}...")

        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get('content-length', 0))
                with open(local_path, 'wb') as f, tqdm(
                    desc=file,
                    total=total,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024
                ) as bar:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
            print(f"✓ Successfully downloaded {file}\n")
        except requests.RequestException as e:
            print(f"✗ Failed to download {file}: {e}\n")

    print("Download complete!")

