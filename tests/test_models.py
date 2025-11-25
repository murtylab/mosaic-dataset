# tests/test_pretrained_models.py
import pytest
import torch
import torch.nn as nn
from mosaic import from_pretrained
from mosaic.utils.json import load_json
from mosaic.constants import subject_id_to_file_mapping

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
valid_params_multihead = [x for x in valid_params_multihead if x[1] == "multihead"]

# ===================================================================
# The actual test
# ===================================================================
@pytest.mark.parametrize(
    "backbone, framework, subjects, vertices", valid_params_multihead
)
def test_pretrained_model_multihead(backbone, framework, subjects, vertices):

    if framework == "multihead":
        all_dataset_names = list(subject_id_to_file_mapping.keys())

        print(f"\nTesting: backbone={backbone} | framework={framework} | subjects='{subjects}' | vertices='{vertices}'")

        """
        corner case for 
        "model-CNN8_framework-multihead_subjects-all_vertices-visual.pth",
        "model-CNN8_framework-multihead_subjects-NSD_vertices-all.pth",
        "model-CNN8_framework-multihead_subjects-NSD_vertices-visual.pth",
        """
        if backbone == "CNN8":
            if framework == "multihead" and subjects == "all":
                assert vertices == "visual"
            elif framework == "multihead" and subjects == "NSD":
                assert vertices in ["all", "visual"]
                all_dataset_names = ["NaturalScenesDataset"]
            

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

        names_and_subjects = {
            dataset_name: "all"
            for dataset_name in all_dataset_names
        }

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(
                x=x,
                names_and_subjects=names_and_subjects
            )

        """
        This is how I expect the output to look like for multihead models:
        {
            dataset_1: {
                subject_1: tensor of shape (batch, num_voxels),
                subject_2: tensor of shape (batch, num_voxels),
                ...
            },
            dataset_2: {
                subject_1: tensor of shape (batch, num_voxels),
                subject_2: tensor of shape (batch, num_voxels),
                ...
            },
            ...
        }
        """

        assert isinstance(output, dict), f"Expected output to be a dict for multihead models, but got: {type(output)}"
        assert list(output.keys()) == all_dataset_names, f"Expected output keys to be {all_dataset_names}, but got: {list(output.keys())}"
        for dataset_name in all_dataset_names:
            single_dataset_output = output[dataset_name]
            assert isinstance(single_dataset_output, dict), f"Expected output[{dataset_name}] to be a dict, but got: {type(single_dataset_output)}"
            num_subjects = len(subject_id_to_file_mapping[dataset_name])
            assert len(single_dataset_output) == num_subjects, f"Expected output[{dataset_name}] to have {num_subjects} subjects, but got: {len(single_dataset_output)}"

            for subject in single_dataset_output:
                assert torch.is_tensor(single_dataset_output[subject]), f"Expected output[{dataset_name}][{subject}] to be a torch tensor, but got: {type(single_dataset_output[subject])}"
                assert single_dataset_output[subject].ndim == 2, f"Expected output[{dataset_name}][{subject}] to be a 2D tensor (batch, num_voxels), but got: {single_dataset_output[subject].ndim}D tensor"
    else:
        raise AssertionError(f"Test received unsupported framework: {framework}")
   

valid_params_singlehead = [parse_singlehead_checkpoint_name(cp) for cp in supported_checkpoints_singlehead]
@pytest.mark.parametrize(
    "backbone, framework, subjects, vertices", valid_params_singlehead
)
def test_pretrained_model_singlehead_cnn8(backbone, framework, subjects, vertices):
    if framework == "singlehead" and backbone == "CNN8":
        print(f"\nTesting: backbone={backbone} | framework={framework} | subjects='{subjects}' | vertices='{vertices}'")

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

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)

        assert torch.is_tensor(output), f"Expected output to be a torch tensor, but got: {type(output)}"
        assert output.ndim == 2, f"Expected output to be a 2D tensor, but got: {output.ndim}D tensor"
        assert output.shape[0] == 1, f"Expected output shape[0] to be 1, but got: {output.shape[0]}"
        assert output.shape[1] == 7831, f"Expected output shape[1] (num voxels) to be 7831, but got: {output.shape[1]}"
