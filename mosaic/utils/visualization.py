import torch
import numpy as np
import hcp_utils as hcp
import nilearn.plotting as plotting
from mosaic.constants import region_of_interest_labels
from IPython.display import HTML

from ..models.transforms import SelectROIs
from ..utils.parcellation import parse_betas

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

def visualize_voxel_data(data: np.ndarray, save_as: str, mode: str, symmetric_cmap: bool, ignore_nan: bool = True) -> None:

    """
    if ignore_nan is True, we will replace NaN values with zeros.
    """
    
    if not ignore_nan:
        if np.isnan(data).any():
            raise ValueError("Data contains NaN values. Please remove or impute them before visualization.")
    else:
        data = np.nan_to_num(data, nan=0.0)
    
    plotting_mode = getattr(hcp.mesh, mode)
    stat = hcp.cortex_data(data)

    if not symmetric_cmap:
        vmin=np.nanmin(stat)
        vmax=np.nanmax(stat)
    else:
        vmin = None
        vmax = None
    html_thing = plotting.view_surf(
        plotting_mode,
        surf_map=stat,
        threshold=None,
        vmin=vmin,
        vmax=vmax,
        bg_map=hcp.mesh.sulc,
        symmetric_cmap=symmetric_cmap
    )
    html_thing.save_as_html(save_as)
    return html_thing


def visualize(
    betas: dict, save_as: str, mode="inflated", rois: list[str] = None, show=True, symmetric_cmap: bool = True, ignore_nan: bool = True
) -> None:
    
    data_to_visualize = np.zeros(len(parcel_map))
    roi_selection = SelectROIs(
        selected_rois="all" if rois is None else rois
    )

    if not isinstance(betas, dict):
        assert isinstance(betas, np.ndarray), f"Expected betas to be a dict or np.ndarray, got {type(betas)} instead"
        betas = parse_betas(betas)

    assert isinstance(
        betas, dict
    ), f"Expected betas to be a dict, got {type(betas)} instead"
    if rois == None:
        rois = [roi for roi in betas.keys() if roi in roi_selection.roi_to_index.keys()]
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

    html_thing = visualize_voxel_data(data=data_to_visualize, save_as=save_as, mode=mode, symmetric_cmap=symmetric_cmap, ignore_nan=ignore_nan)

    if show:
        return html_thing
    else:
        return None