import os
import h5py
import numpy as np
from ..utils.download import download_file
from ..utils.folder import make_folder_if_does_not_exist
from ..constants import BASE_URL
from ..utils.parcellation import parse_betas
from tqdm import tqdm

subject_id_to_file_mapping = {
    1: "sub-01_NSD_crc32-32353cf9.hdf5",
    2: "sub-02_NSD_crc32-32353cf9.hdf5",
}

class NSDSingleSubject:
    def __init__(
        self,
        folder: str,
        subject_id: int = 1,
        cache: bool = True,
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

        if cache:
            self.data = {
                'betas': {
                    name: np.array(self.data['betas'][name]) for name in tqdm(self.all_names, desc="Caching betas")
                }
            }
        else:
            self.data = self.data

    def __getitem__(self, index: int) -> dict:
        
        item = np.array(
            self.data['betas'][self.all_names[index]]
        )
        item = parse_betas(betas=item)
        return {
            'name': self.all_names[index],
            'betas': item,
        }
    
    def __len__(self) -> int:
        return len(self.all_names)
