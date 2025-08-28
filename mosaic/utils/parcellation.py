import hcp_utils as hcp
import numpy as np

"""
Reference: https://link.springer.com/article/10.1007/s00429-021-02421-6

PIT = posterior inferotemporal cortex
FFC = fusiform face complex (faces and a little bit of bodies)
PHA = parahippocampal area (scenes)
LO = lateral occipital cortex (bodies)


for other labels, run this snippet

```
import hcp_utils as hcp

parcellation  = hcp.mmp
from pprint import pprint
pprint(hcp.mmp.labels)
```
"""

regions_of_interest_labels = {
    "left":{
        "V1": 1,
        "V2": 4,
        "V3": 5,
        "V4": 6,
        "PIT": 22,
        "FFC": 18,
        "PHA1": 126,
        "PHA2": 155,
        "PHA3": 127,
        "LO1": 20,
        "LO2": 21
    },
    "right":{
        "V1": 181,
        "V2": 184,
        "V3": 185,
        "V4": 186,
        "PIT": 202,
        "FFC": 198,
        "PHA1": 306,
        "PHA2": 335,
        "PHA3": 307,
        "LO1": 200,
        "LO2": 201
    }
}

parcellation  = hcp.mmp

def parse_betas(betas: np.ndarray):
    parcel_map = parcellation.map_all

    result = {}
    for hemisphere in ["left", "right"]:
        result[hemisphere] = {}
        for region in regions_of_interest_labels[hemisphere].keys():
            label = regions_of_interest_labels[hemisphere][region]
            region_data = betas[parcel_map == label]
            result[hemisphere][region] = region_data

    return result