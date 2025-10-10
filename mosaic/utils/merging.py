import h5py
import numpy as np
import os

def merge_hdf5_files(files: list[str], save_as: str) -> None:
    assert len(files) > 1, "Need at least two files to merge."

    merged_data = {}

    for index, filename in enumerate(files):
        assert os.path.exists(filename), f"File {filename} does not exist."
        with h5py.File(filename, "r") as data:
            if index == 0:
                # Initialize merged_data with keys and arrays
                merged_data["betas"] = {key: np.array(data["betas"][key]) for key in data["betas"].keys()}
            else:
                # Merge each dataset under "betas"
                for key in data["betas"].keys():
                    if key in merged_data["betas"]:
                        merged_data["betas"][key] = np.concatenate(
                            (merged_data["betas"][key], np.array(data["betas"][key])), axis=0
                        )
                    else:
                        merged_data["betas"][key] = np.array(data["betas"][key])

    # Save merged data
    with h5py.File(save_as, "w") as f:
        betas_group = f.create_group("betas")
        for key, array in merged_data["betas"].items():
            betas_group.create_dataset(key, data=array)

    print(f"Merged file saved as {save_as}")
