import torch
import numpy as np
import hcp_utils as hcp
import nilearn.plotting as plotting
from mosaic.constants import region_of_interest_labels
from IPython.display import HTML

valid_modes = [
    "white",
    "midthickness",
    "pial",
    "inflated",
    "very_inflated",
    "flat",
    "sphere",
]
valid_rois = list(region_of_interest_labels.keys())

parcellation = hcp.mmp
parcel_map = parcellation.map_all

def render_html_in_notebook(filename: str):
    with open(filename, "r") as f:
        html = f.read()

    return HTML(html)

def visualize_voxel_data(data: np.ndarray, save_as: str, mode: str) -> None:
    plotting_mode = getattr(hcp.mesh, mode)
    data[np.isnan(data)] = 0
    html_thing = plotting.view_surf(
        plotting_mode,
        surf_map=hcp.cortex_data(data),
        threshold=0.0,
        bg_map=hcp.mesh.sulc,
    )
    html_thing.save_as_html(save_as)


def visualize(
    betas: dict, save_as: str, mode="inflated", rois: list[str] = None, show=True
) -> None:

    assert isinstance(
        betas, dict
    ), f"Expected betas to be a dict, got {type(betas)} instead"

    data_to_visualize = np.zeros(len(parcel_map))

    if rois is None:
        rois = list(betas.keys())
    else:
        for roi in rois:
            assert (
                roi in valid_rois
            ), f"Invalid roi: {roi}\n Expected it to be one of: {valid_rois}"

    for roi in rois:
        data_to_visualize[parcel_map == region_of_interest_labels[roi]] = betas[roi]

    assert (
        mode in valid_modes
    ), f"Expected mode to be one of {valid_modes}, got {mode} instead"

    visualize_voxel_data(data=data_to_visualize, save_as=save_as, mode=mode)

    if show:
        return render_html_in_notebook(filename=save_as)
    else:
        return None