from typing import Union
from .pretrained import valid_backbone_names, valid_vertices, get_pretrained_backbone
import torch.nn as nn


def from_pretrained(
    backbone_name: str = "ResNet18",
    framework: str = "multihead",
    subjects: Union[str, list] = "all",
    vertices: str = "visual",
    folder: str = "./mosaic_models/",
    pretrained: bool = True, ## set this to false if you want an untrained model
) -> nn.Module:
    """
    Download and load a pretrained brain optimized model by specifying backbone, framework, subjects, and vertices

    INPUTS:
    backbone_name: str, name of backbone (core) model, or core model. 
    framework: str, name of model framework readouts e.g., multihead or singlehead
    subjects: str or list[str], the subjects that the model was trained on.
    vertices: str, the vertices that model was trained on. visual --> MMP1.0 ROI sections 1-5, all --> MMP1.0 ROI sections 1-22
    folder: local folder to download the checkpoint to

    RETURNS:
    pretrained backbone, type torch module in nn.DataParallel. specified architecture with pretrained weights loaded in and set to eval mode.
    
    """
    assert (
        backbone_name in valid_backbone_names
    ), f"Invalid backbone_name {backbone_name}. Must be one of {valid_backbone_names}"
    assert (
        vertices in valid_vertices[backbone_name]
    ), f"Invalid vertices: {vertices}. Must be one of: {valid_vertices[backbone_name]}"


    model = get_pretrained_backbone(
        backbone_name=backbone_name,
        framework=framework,
        subjects=subjects,
        vertices=vertices,
        folder=folder,
        pretrained=pretrained,
    )

    model.vertices = vertices

    return model.eval()