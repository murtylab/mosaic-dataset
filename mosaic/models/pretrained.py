import os
import torch
from ..constants import BASE_URL
from ..utils.download import download_file
from .transforms import SelectROIs
from .readout import SpatialXFeatureLinear
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
    "CNN8": ["model-CNN8_framework-multihead_subjects-all_vertices-visual.pth",
             "model-CNN8_framework-multihead_subjects-NSD_vertices-all.pth",
             "model-CNN8_framework-multihead_subjects-NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-all_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-01_deeprecon_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-01_NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-02_deeprecon_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-02_NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-03_deeprecon_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-03_NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-04_NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-05_NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-06_NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-07_NSD_vertices-visual.pth",
             "model-CNN8_framework-singlehead_subjects-sub-08_NSD_vertices-visual.pth"]
}
supported_checkpoints_list = [item for sublist in supported_checkpoints.values() for item in sublist]

valid_subjects = {
    "AlexNet": ["all"],
    "ResNet18": ["all"],
    "SqueezeNet1_1": ["all"],
    "SwinT": ["all"],
    "CNN8": ['all',
             "NSD",
             "sub-01_deeprecon",
             "sub-02_deeprecon",
             "sub-03_deeprecon",
             "sub-01_NSD",
             "sub-02_NSD",
             "sub-03_NSD",
             "sub-04_NSD",
             "sub-05_NSD",
             "sub-06_NSD",
             "sub-07_NSD",
             "sub-08_NSD"]
}

from .architectures import (
    AlexNetCore,
    ResNet18Core,
    SqueezeNet1_1Core,
    SwinTCore,
    Encoder,
    EncoderMultiHead,
    C8NonSteerableCNN
)
from typing import Union
import requests


def get_pretrained_backbone(
    backbone_name: str,
    framework: str,
    subjects: Union[str, list],
    vertices: str,
    folder: str = "./mosaic_models/",
    device="cpu",
):
    if not os.path.exists(path=folder):
        os.mkdir(folder)

    if "AlexNet" == backbone_name:
        bo_core = AlexNetCore(add_batchnorm=True)  # brain optimized pretrained
    elif "ResNet18" == backbone_name:
        bo_core = ResNet18Core()
    elif "SqueezeNet1_1" == backbone_name:
        bo_core = SqueezeNet1_1Core(add_batchnorm=True)
    elif "SwinT" == backbone_name:
        bo_core = SwinTCore()
    elif "CNN8" == backbone_name:
        bo_core = C8NonSteerableCNN()
    else:
        raise ValueError(
            f"Invalid backbone_name {backbone_name}. Must be one of {valid_backbone_names}"
        )

    # get the correct number of output vertices
    if vertices == "visual":
        rois = [f"GlasserGroup_{x}" for x in range(1, 6)]
    elif vertices == "all":
        rois = [f"GlasserGroup_{x}" for x in range(1, 23)]

    desired_checkpoint = f"model-{backbone_name}_framework-{framework}_subjects-{subjects}_vertices-{vertices}.pth"

    assert desired_checkpoint in supported_checkpoints_list, f"Your specified checkpoint {desired_checkpoint} is not yet supported or does not exist. Must be one of {supported_checkpoints_list}."

    desired_checkpoint_local_path = os.path.join(folder, desired_checkpoint)
    #if you haven't downloaded the checkpoint yet, download it. otherwise load it from your local folder
    if not os.path.exists(desired_checkpoint_local_path):
        url = BASE_URL + "/" + model_folder_s3 + "/" + desired_checkpoint
        response = requests.head(url)
        assert response.status_code == 200, f"URL {url} is not valid or reachable."
        download_file(
            base_url=BASE_URL,
            file=model_folder_s3 + "/" + desired_checkpoint,
            save_as=desired_checkpoint_local_path,
        )
    else:
        print(f"Checkpoint {desired_checkpoint_local_path} already downloaded.")

    ROI_selection = SelectROIs(
        selected_rois=rois,
    )
    num_vertices = len(ROI_selection.selected_roi_indices)
    # print(f"number of vertices/regression targets: {num_vertices}")
    with torch.no_grad():
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
    state_dict = torch.load(desired_checkpoint_local_path, map_location="cpu")
    bo_model.load_state_dict(state_dict, strict=True)
    bo_model = bo_model.eval()
    model = bo_model.module #return the underlying model, not the dataparallel wrapper
    model.vertices = vertices
    return (
        model
    )  
