import os
import h5py
import numpy as np
from ..utils.download import download_file
from ..utils.folder import make_folder_if_does_not_exist
from ..constants import BASE_URL

subject_id_to_file_mapping = {
    1: "sub-01_BOLD5000_crc32-4d74c3d3.hdf5",
    2: "sub-02_BOLD5000_crc32-dac70140.hdf5",
}

class BOLD5000SingleSubject:
    def __init__(
        self,
        folder: str,
        subject_id: int = 1,
    ): 
        assert subject_id in list(subject_id_to_file_mapping.keys()), \
            f"Subject ID {subject_id} is not valid. Please choose from {list(subject_id_to_file_mapping.keys())}."
        
        self.folder = folder
        self.subject_id = subject_id
        self.filename = os.path.join(self.folder, subject_id_to_file_mapping[self.subject_id])

        if not os.path.exists(self.filename):
            make_folder_if_does_not_exist(folder=self.folder)
            download_file(
                base_url=BASE_URL,
                file=subject_id_to_file_mapping[self.subject_id],
                save_as=self.filename,
            )

        assert os.path.exists(self.filename), f"File {self.filename} does not exist. Please check the download."
        ## the filename is an hdf5 file
        self.data = h5py.File(self.filename, 'r')
        self.all_names = list(self.data['betas'].keys())

    def __getitem__(self, index: int) -> dict:
        
        item = np.array(
            self.data['betas'][self.all_names[index]]
        )
        return {
            'name': self.all_names[index],
            'betas': item,
        }
    
    def __len__(self) -> int:
        return len(self.all_names)
