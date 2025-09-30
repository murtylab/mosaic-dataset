from typing import Union, List
from ..datasets import (
    BOLD5000SingleSubject,
    GenericObjectDecodingSingleSubject,
    NSDSingleSubject,
    ThingsFMRISingleSubject,
    DeepReconSingleSubject,
)


class MultiSubjectDataset:
    def __init__(
        self,
        datasets: Union[
            List[BOLD5000SingleSubject],
            List[GenericObjectDecodingSingleSubject],
            List[NSDSingleSubject],
            List[ThingsFMRISingleSubject],
            List[DeepReconSingleSubject],
        ],
    ):
        self.datasets = datasets
        self.index_to_subject_index_mapping = []

        for subject_index, d in enumerate(datasets):
            self.index_to_subject_index_mapping.extend([subject_index] * len(d))

    def __getitem__(self, index: int) -> dict:
        subject_index = self.index_to_subject_index_mapping[index]
        index_inside_subject = index - sum(
            len(self.datasets[i]) for i in range(subject_index)
        )
        item = self.datasets[subject_index][index_inside_subject]
        return {
            "name": item["name"],
            "betas": item["betas"],
            "subject_index": subject_index,
        }

    def __len__(self) -> int:
        return sum(len(d) for d in self.datasets)
