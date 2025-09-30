import hcp_utils as hcp
import numpy as np
from ..constants import region_of_interest_labels

"""
Reference: https://link.springer.com/article/10.1007/s00429-021-02421-6
"""

parcellation = hcp.mmp
parcel_map = parcellation.map_all


def parse_betas(betas: np.ndarray):

    result = {}
    for region in region_of_interest_labels.keys():
        label = region_of_interest_labels[region]
        region_data = betas[parcel_map == label]
        result[region] = region_data

    return result
