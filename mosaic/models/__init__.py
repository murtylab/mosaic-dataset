from typing import Union
from .pretrained import valid_backbone_names, valid_vertices, get_pretrained_backbone
import torch.nn as nn


def from_pretrained(
    backbone_name: str = "resnet18",
    vertices: Union[str, list] = "visual",
    framework: str = "multihead",
    subjects: Union[str, list] = "all",
    folder: str = "./mosaic_models/",
) -> nn.Module:
    assert (
        backbone_name in valid_backbone_names
    ), f"Invalid backbone_name {backbone_name}. Must be one of {valid_backbone_names}"
    assert (
        vertices in valid_vertices
    ), f"Invalid vertices: {vertices}. Must be one of: {valid_vertices}"

    return get_pretrained_backbone(
        backbone_name=backbone_name,
        vertices=vertices,
        framework=framework,
        subjects=subjects,
        folder=folder,
    )
