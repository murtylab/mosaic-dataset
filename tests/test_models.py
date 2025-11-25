# tests/test_pretrained_models.py
import pytest
import torch
import torch.nn as nn
from mosaic import from_pretrained
from mosaic.utils.json import load_json

testing_config = load_json("tests/testing_config.json")

MODEL_FOLDER = testing_config["models_folder"]
DOWNLOAD_PRETRAINED_MODELS = testing_config["download_pretrained_models"]
# ===================================================================
# Exact list of released checkpoints (copy-pasted from your code)
# ===================================================================
supported_checkpoints_multihead = [
    "model-AlexNet_framework-multihead_subjects-all_vertices-visual.pth",
    "model-ResNet18_framework-multihead_subjects-all_vertices-visual.pth",
    "model-SqueezeNet1_1_framework-multihead_subjects-all_vertices-visual.pth",
    "model-SwinT_framework-multihead_subjects-all_vertices-visual.pth",
    "model-CNN8_framework-multihead_subjects-all_vertices-visual.pth",
    "model-CNN8_framework-multihead_subjects-NSD_vertices-all.pth",
    "model-CNN8_framework-multihead_subjects-NSD_vertices-visual.pth",
]

# ===================================================================
# Robust parser that never breaks subject strings
# ===================================================================
def parse_multihead_checkpoint_name(filename: str):
    """Parse filename → (backbone, framework, subjects, vertices)"""
    name = filename.replace(".pth", "")
    parts = {}
    for p in name.split("_"):
        if "-" in p:
            k, v = p.split("-", 1)
            parts[k] = v

    backbone = parts["model"]
    # Special alias: allow SqueezeNet1 → SqueezeNet1_1
    if backbone == "SqueezeNet1":
        backbone = "SqueezeNet1_1"

    return (
        backbone,
        parts["framework"],
        parts["subjects"],   # "all", "NSD", "sub-01_NSD", "sub-01_deeprecon"
        parts["vertices"],
    )


supported_checkpoints_singlehead = [
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

def parse_singlehead_checkpoint_name(
    filename: str,
):
    """Parse filename → (backbone, framework, subjects, vertices)"""
    name = filename.replace(".pth", "")
    parts = {}
    for p in name.split("_"):
        if "-" in p:
            k, v = p.split("-", 1)
            parts[k] = v

            if k == "subjects":
                # Re-join subject IDs that may contain hyphens
                if parts[k].startswith("sub"):
                    parts[k] = parts[k] + "_NSD"

    backbone = parts["model"]
    # Special alias: allow SqueezeNet1 → SqueezeNet1_1
    if backbone == "SqueezeNet1":
        backbone = "SqueezeNet1_1"

    return (
        backbone,
        parts["framework"],
        parts["subjects"],   # "all", "NSD", "sub-01_NSD", "sub-01_deeprecon"
        parts["vertices"],
    )


# Generate valid parameter combinations
valid_params_multihead = [parse_multihead_checkpoint_name(cp) for cp in supported_checkpoints_multihead]


# ===================================================================
# The actual test
# ===================================================================
@pytest.mark.parametrize(
    "backbone, framework, subjects, vertices", valid_params_multihead
)
def test_pretrained_model_multihead(backbone, framework, subjects, vertices):

    if framework == "multihead":
        print(f"\nTesting: {backbone} | {framework} | subjects='{subjects}' | vertices='{vertices}'")

        model = from_pretrained(
            backbone_name=backbone,
            framework=framework,
            subjects=subjects,      # pass exactly as in filename
            vertices=vertices,
            folder=MODEL_FOLDER,
            pretrained=DOWNLOAD_PRETRAINED_MODELS
        )

        assert isinstance(model, nn.Module)
        assert model.vertices == vertices

   

valid_params_singlehead = [parse_singlehead_checkpoint_name(cp) for cp in supported_checkpoints_singlehead]
@pytest.mark.parametrize(
    "backbone, framework, subjects, vertices", valid_params_singlehead
)
def test_pretrained_model_singlehead_cnn8(backbone, framework, subjects, vertices):
    if framework == "singlehead" and backbone == "CNN8":
        print(f"\nTesting: {backbone} | {framework} | subjects='{subjects}' | vertices='{vertices}'")

        model = from_pretrained(
            backbone_name=backbone,
            framework=framework,
            subjects=subjects,
            vertices=vertices,
            folder=MODEL_FOLDER,
            pretrained=DOWNLOAD_PRETRAINED_MODELS
        )

        assert isinstance(model, nn.Module)
        assert model.vertices == vertices