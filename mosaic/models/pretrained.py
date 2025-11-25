import os
import torch
import requests
import torch.nn as nn
from typing import Union
from ..constants import BASE_URL
from ..utils.download import download_file
from .transforms import SelectROIs
from .readout import SpatialXFeatureLinear
from ..utils.checkpoint_conversion import convert_dataparallel_state_dict_to_vanilla
import torch.nn as nn

valid_backbone_names = ["AlexNet", "ResNet18", "SqueezeNet1_1", "SwinT", "CNN8"]

valid_vertices = {
    "AlexNet": ["visual"],
    "ResNet18": ["visual"],
    "SqueezeNet1_1": ["visual"],
    "SwinT": ["visual"],
    "CNN8": ["visual", "all"],
}

valid_frameworks = {
    "AlexNet": ["multihead"],
    "ResNet18": ["multihead"],
    "SqueezeNet1_1": ["multihead"],
    "SwinT": ["multihead"],
    "CNN8": ["multihead", "singlehead"],
}

model_folder_s3 = "brain_optimized_checkpoints"

supported_checkpoints = {
    "AlexNet": ["model-AlexNet_framework-multihead_subjects-all_vertices-visual.pth"],
    "ResNet18": ["model-ResNet18_framework-multihead_subjects-all_vertices-visual.pth"],
    "SqueezeNet1_1": ["model-SqueezeNet1_1_framework-multihead_subjects-all_vertices-visual.pth"],
    "SwinT": ["model-SwinT_framework-multihead_subjects-all_vertices-visual.pth"],
    "CNN8": [
        "model-CNN8_framework-multihead_subjects-all_vertices-visual.pth",
        "model-CNN8_framework-multihead_subjects-NSD_vertices-all.pth",
        "model-CNN8_framework-multihead_subjects-NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-all_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-01_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-02_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-03_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-04_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-05_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-06_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-07_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-08_NSD_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-01_deeprecon_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-02_deeprecon_vertices-visual.pth",
        "model-CNN8_framework-singlehead_subjects-sub-03_deeprecon_vertices-visual.pth",
        ]
}

supported_checkpoints_list = [
    item for sublist in supported_checkpoints.values() for item in sublist
]

# Import architectures
from .architectures import (
    AlexNetCore,
    ResNet18Core,
    SqueezeNet1_1Core,
    SwinTCore,
    Encoder,
    EncoderMultiHead,
    C8NonSteerableCNN
)


def get_pretrained_backbone(
    backbone_name: str,
    framework: str,
    subjects: Union[str, list],
    vertices: str,
    folder: str = "./mosaic_models/",
    device: str = "cpu",
    pretrained = True,
):
    """
    Load a brain-optimized pretrained model.

    Important constraints:
    - Only CNN8 has `vertices="all"` models, and only when `subjects="NSD"`
    - `subjects="all"` + `vertices="all"` is not supported
    - For single-subject models, `subjects` must be a string like "sub-01_NSD"
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if backbone_name not in valid_backbone_names:
        raise ValueError(f"backbone_name must be one of {valid_backbone_names}")

    if framework not in valid_frameworks[backbone_name]:
        raise ValueError(f"For {backbone_name}, framework must be one of {valid_frameworks[backbone_name]}")

    if vertices not in valid_vertices[backbone_name]:
        raise ValueError(f"For {backbone_name}, vertices must be one of {valid_vertices[backbone_name]}")

    # ------------------------------------------------------------------
    # Normalize subjects → string
    # ------------------------------------------------------------------
    if isinstance(subjects, list):
        if len(subjects) != 1:
            raise ValueError("If subjects is a list, it must contain exactly one subject ID.")
        subjects_str = subjects[0]
    else:
        subjects_str = subjects

    # ------------------------------------------------------------------
    # Block invalid combinations early (before filename construction)
    # ------------------------------------------------------------------
    if vertices == "all" and subjects_str != "NSD":
        raise ValueError(
            "vertices='all' is only supported for subjects='NSD' (whole-cortex NSD model). "
            "No other combination exists."
        )

    if subjects_str == "all" and vertices == "all":
        raise ValueError(
            "Combination subjects='all' and vertices='all' does not exist. "
            "Use subjects='NSD' + vertices='all' for whole-cortex prediction."
        )

    # ------------------------------------------------------------------
    # Construct checkpoint filename
    # ------------------------------------------------------------------
    desired_checkpoint = (
        f"model-{backbone_name}_framework-{framework}_subjects-{subjects_str}_vertices-{vertices}.pth"
    )

    if desired_checkpoint not in supported_checkpoints_list:
        raise AssertionError(
            f"\nCheckpoint not found: {desired_checkpoint}\n\n"
            "This usually happens because:\n"
            "  • You requested vertices='all' with subjects ≠ 'NSD'\n"
            "  • You requested subjects='all' + vertices='all'\n"
            "  • The specific single-subject + framework combo was never released\n\n"
            "Supported checkpoints:\n" + "\n".join(f"  • {cp}" for cp in supported_checkpoints_list)
        )

    # ------------------------------------------------------------------
    # Download if needed
    # ------------------------------------------------------------------
    if pretrained:
        os.makedirs(folder, exist_ok=True)
        local_path = os.path.join(folder, desired_checkpoint)

        if not os.path.exists(local_path):
            url = f"{BASE_URL}/{model_folder_s3}/{desired_checkpoint}"
            response = requests.head(url)
            if response.status_code != 200:
                raise RuntimeError(f"Checkpoint not reachable: {url}")
            print(f"Downloading {desired_checkpoint} → {local_path}")
            download_file(
                base_url=BASE_URL,
                file=f"{model_folder_s3}/{desired_checkpoint}",
                save_as=local_path,
            )
        else:
            print(f"Using cached checkpoint: {local_path}")
    else:
        pass
    # ------------------------------------------------------------------
    # Build core
    # ------------------------------------------------------------------
    if backbone_name == "AlexNet":
        bo_core = AlexNetCore(add_batchnorm=True)
    elif backbone_name == "ResNet18":
        bo_core = ResNet18Core()
    elif backbone_name == "SqueezeNet1_1":
        bo_core = SqueezeNet1_1Core(add_batchnorm=True)
    elif backbone_name == "SwinT":
        bo_core = SwinTCore()
    elif backbone_name == "CNN8":
        bo_core = C8NonSteerableCNN()
    else:
        raise RuntimeError(f"Invalid backbone_name: {backbone_name}")

    # ------------------------------------------------------------------
    # ROI selection & readout setup
    # ------------------------------------------------------------------
    if vertices == "visual":
        rois = [f"GlasserGroup_{i}" for i in range(1, 6)]
    elif vertices == "all":
        rois = [f"GlasserGroup_{i}" for i in range(1, 23)]
    else:
        raise ValueError("vertices must be 'visual' or 'all'")

    roi_selector = SelectROIs(selected_rois=rois)
    num_vertices = len(roi_selector.selected_roi_indices)

    with torch.no_grad():
        feature_shape = bo_core(torch.randn(1, 3, 224, 224)).shape[1:]

    readout_kwargs = {
        "in_shape": feature_shape,
        "outdims": num_vertices,
        "bias": True,
        "normalize": True,
        "init_noise": 1e-3,
        "constrain_pos": False,
        "positive_weights": False,
        "positive_spatial": False,
    }

    # ------------------------------------------------------------------
    # Build full model
    # ------------------------------------------------------------------
    if framework == "singlehead":
        readout = SpatialXFeatureLinear(**readout_kwargs)
        model = Encoder(bo_core, readout).to(device)

    elif framework == "multihead":
        # Build subject index mapping (same logic as training)
        numsubs = {
            "NSD": 8,
            "BOLD5000": 4,
            "BMD": 10,
            "THINGS": 3,
            "NOD": 30,
            "HAD": 30,
            "GOD": 5,
            "deeprecon": 3,
        }

        if subjects_str == "all":
            training_subjects = [
                f"sub-{i:02d}_{dset}" for dset, n in numsubs.items() for i in range(1, n + 1)
            ]
        elif subjects_str in numsubs:
            n = numsubs[subjects_str]
            training_subjects = [f"sub-{i:02d}_{subjects_str}" for i in range(1, n + 1)]
        else:
            training_subjects = [subjects_str]

        # Sort exactly like during training
        training_subjects = sorted(
            training_subjects,
            key=lambda x: (x.split("_")[1], int(x.split("-")[1].split("_")[0])),
        )
        subjectID2idx = {sid: idx for idx, sid in enumerate(training_subjects)}

        model = EncoderMultiHead(
            core=bo_core,
            readout_class=SpatialXFeatureLinear,
            subjectID2idx=subjectID2idx,
            **readout_kwargs,
        ).to(device)

        # Required because models were saved with DataParallel
        model = nn.DataParallel(model)

    else:
        raise ValueError(f"Unknown framework: {framework}")

    if pretrained:
        # ------------------------------------------------------------------
        # Load weights
        # ------------------------------------------------------------------
        state_dict = torch.load(local_path, map_location="cpu")

        if not isinstance(model, nn.DataParallel):
            state_dict = convert_dataparallel_state_dict_to_vanilla(state_dict)

        model.load_state_dict(state_dict, strict=True)
    else:
        pass

    model = model.eval()

    # Remove DataParallel wrapper (if present)
    if isinstance(model, nn.DataParallel):
        model = model.module

    model.vertices = vertices  # useful metadata
    return model