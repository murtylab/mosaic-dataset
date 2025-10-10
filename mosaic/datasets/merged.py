import os
import h5py
import numpy as np
from ..utils.parcellation import parse_betas


class MergedDataset:
    def __init__(
        self,
        filename: str
    ):
        self.filename = filename
        assert os.path.exists(
            self.filename
        ), f"File {self.filename} does not exist. Please check the download."
        ## the filename is an hdf5 file
        try:
            self.data = h5py.File(self.filename, "r")
        except OSError:
            raise OSError(
                f"Could not read file: {self.filename}. Maybe just it's a failed download?"
            )
        self.all_names = list(self.data["betas"].keys())

    def __getitem__(self, index: int) -> dict:

        item = np.array(self.data["betas"][self.all_names[index]])
        item = parse_betas(betas=item)
        return {
            "name": self.all_names[index],
            "betas": item,
        }

    def __len__(self) -> int:
        return len(self.all_names)
