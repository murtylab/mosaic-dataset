import os
import torch
from ..constants import BASE_URL
from ..utils.download import download_file
from .transforms import SelectROIs
from .readout import SpatialXFeatureLinear
import torch.nn as nn

valid_backbone_names = ["alexnet", "resnet18", "squeezenet", "swint"]
valid_vertices = ["visual"]
valid_frameworks = ["multihead"]

model_folder = "brain_optimized_checkpoints"

model_filenames = {
    "alexnet": os.path.join(
        model_folder,
        "model-AlexNet_framework-multihead_subjects-all_vertices-visual.pth",
    ),
    "resnet18": os.path.join(
        model_folder,
        "model-ResNet18_framework-multihead_subjects-all_vertices-visual.pth",
    ),
    "squeezenet": os.path.join(
        model_folder,
        "model-SqueezeNet1_1_framework-multihead_subjects-all_vertices-visual.pth",
    ),
    "swint": os.path.join(
        model_folder, "model-SwinT_framework-multihead_subjects-all_vertices-visual.pth"
    ),
}

from .architectures import (
    AlexNetCore,
    ResNet18Core,
    SqueezeNet1_1Core,
    SwinTCore,
    Encoder,
    EncoderMultiHead,
)
from typing import Union
import requests


def get_pretrained_backbone(
    backbone_name: str,
    vertices: str,
    framework: str,
    subjects: Union[str, list],
    folder: str = "./mosaic_models/",
    device="cpu",
):
    if not os.path.exists(path=folder):
        os.mkdir(folder)

    if "alexnet" == backbone_name:
        bo_core = AlexNetCore(add_batchnorm=True)  # brain optimized pretrained
    elif "resnet18" == backbone_name:
        bo_core = ResNet18Core()
    elif "squeezenet" == backbone_name:
        bo_core = SqueezeNet1_1Core(add_batchnorm=True)
    elif "swint" == backbone_name:
        bo_core = SwinTCore()
    # elif "CNN8" == backbone_name:
    #     bo_core = C8NonSteerableCNN()
    else:
        raise ValueError(
            f"Invalid backbone_name {backbone_name}. Must be one of {valid_backbone_names}"
        )

    # get the correct number of output vertices
    if vertices == "visual":
        rois = [f"GlasserGroup_{x}" for x in range(1, 6)]
    elif vertices == "all":
        rois = [f"GlasserGroup_{x}" for x in range(1, 23)]

    # print(
    #     f"Loading pretrained backbone: {backbone_name} vertices: {vertices} framework: {framework} subjects: {subjects}"
    # )

    checkpoint_filename = os.path.join(
        folder,
        f"model-{backbone_name}_framework-{framework}_subjects-{subjects}_vertices-{vertices}.pth",
    )

    if not os.path.exists(checkpoint_filename):
        url = BASE_URL + "/" + model_filenames[backbone_name]
        response = requests.head(url)
        assert response.status_code == 200, f"URL {url} is not valid or reachable."
        download_file(
            base_url=BASE_URL,
            file=model_filenames[backbone_name],
            save_as=checkpoint_filename,
        )

    ROI_selection = SelectROIs(
        selected_rois=rois,
    )
    num_vertices = len(ROI_selection.selected_roi_indices)
    # print(f"number of vertices/regression targets: {num_vertices}")

    out_shape = bo_core(torch.randn(1, 3, 224, 224)).size()[1:]
    readout_kwargs = {
        "in_shape": out_shape,
        "bias": True,
        "normalize": True,
        "init_noise": 1e-3,
        "constrain_pos": False,
        "positive_weights": False,
        "positive_spatial": False,
        "outdims": num_vertices,
    }
    if framework == "singlehead":
        # subjects doesn't affect the initialization of single head
        readout = SpatialXFeatureLinear(
            in_shape=readout_kwargs["in_shape"],
            outdims=readout_kwargs["outdims"],
            init_noise=readout_kwargs["init_noise"],
            normalize=readout_kwargs["normalize"],
            constrain_pos=readout_kwargs["constrain_pos"],
            bias=readout_kwargs["bias"],
        )
        bo_model = Encoder(bo_core, readout).to(device)
    elif framework == "multihead":
        # get the correct number of prediction subjects
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
        training_subjects = []
        if subjects == "all":
            for dset, nsubs in numsubs.items():
                training_subjects += [
                    f"sub-{x:02}_{dset}" for x in range(1, numsubs[dset] + 1)
                ]

        elif subjects in list(
            numsubs.keys()
        ):  # user specified a dataset, meaning all subjects in this dataset
            training_subjects = [
                f"sub-{x:02}_{subjects}" for x in range(1, numsubs[subjects] + 1)
            ]
        else:
            training_subjects = subjects  # just one individual subject specified
        training_subjects_sorted = sorted(
            training_subjects,
            key=lambda x: (x.split("_")[1], int(x.split("_")[0].split("-")[-1])),
        )  # this is how the brain optimized model sorted them

        subjectID2idx = {
            subjectID: idx for idx, subjectID in enumerate(training_subjects_sorted)
        }
        bo_model = EncoderMultiHead(
            bo_core,
            SpatialXFeatureLinear,
            subjectID2idx=subjectID2idx,
            **readout_kwargs,
        ).to(device)

        bo_model = nn.DataParallel(
            bo_model
        )  # must use dataparallel because this is how the model was trained and weights saved
        state_dict = torch.load(checkpoint_filename)
        bo_model.load_state_dict(state_dict, strict=True)
        bo_model = bo_model.eval()
        return (
            bo_model.module
        )  # return the underlying model, not the dataparallel wrapper
