import torch
import numpy as np
import hcp_utils as hcp
import nilearn.plotting as plotting
from mosaic.constants import region_of_interest_labels
from IPython.display import HTML

from mosaic.models.transforms import SelectROIs

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
    stat = hcp.cortex_data(data)
    vmin=np.nanmin(stat)
    vmax=np.nanmax(stat)
    html_thing = plotting.view_surf(
        plotting_mode,
        surf_map=stat,
        threshold=None,
        vmin=vmin,
        vmax=vmax,
        bg_map=hcp.mesh.sulc,
        symmetric_cmap=False
    )
    html_thing.save_as_html(save_as)
    return html_thing


def visualize(
    betas: dict, save_as: str, mode="inflated", rois: list[str] = None, show=True
) -> None:
    
    data_to_visualize = np.zeros(len(parcel_map))
    roi_selection = SelectROIs(
        selected_rois="all" if rois is None else rois
    )

    assert isinstance(
        betas, dict
    ), f"Expected betas to be a dict, got {type(betas)} instead"
    if rois == None:
        rois = list(betas.keys())
    else:
        for roi in rois:
            assert (
                roi in valid_rois
            ), f"Invalid roi: {roi}\n Expected it to be one of: {valid_rois}"

    for roi in rois:
        if len(roi) == 0:
            continue
        try:
            data_to_visualize[roi_selection.roi_to_index[roi]] = betas[roi]
        except KeyError:
            print(f"Warning: ROI {roi} not found in betas dictionary. Skipping.")

    assert (
        mode in valid_modes
    ), f"Expected mode to be one of {valid_modes}, got {mode} instead"

    html_thing = visualize_voxel_data(data=data_to_visualize, save_as=save_as, mode=mode)

    if show:
        return html_thing
    else:
        return None