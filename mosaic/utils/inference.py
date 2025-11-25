import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.transforms as transforms
import hcp_utils as hcp
import nilearn.plotting as plotting
from ..models.transforms import SelectROIs
from ..constants import num_subjects
from ..models.architectures import C8NonSteerableCNN
import warnings
from typing import Union

imagenet_transforms = transforms.Compose(
    [
        transforms.Lambda(lambda img: transforms.CenterCrop(min(img.size))(img)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

valid_plot_modes = ['white', 'midthickness', 'pial', 'inflated', 'very_inflated', 'flat', 'sphere']

def check_if_single_subject_model(model):
    assert hasattr(model, "framework"), f"Model must have a 'framework' attribute, but got this model instead:\n {model}"
    return model.framework == "singlehead"

class MosaicInference:
    def __init__(
        self,
        model,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device).eval()
        self.batch_size = batch_size
        self.device = device
        self.model.eval()
        assert hasattr(model, "vertices"), "Model must have a 'vertices' attribute indicating the vertices it was trained on. If using from_pretrained, this is automatically added."
        self.is_single_subject_model = check_if_single_subject_model(model)

    def run_multi_subject_inference(
        self,
        images: torch.Tensor,
        names_and_subjects: dict,
    ) -> dict:
        results = []
        # Handles the last batch even if it's smaller than batch_size
        for i in tqdm(range(0, len(images), self.batch_size), total=len(images)//self.batch_size + 1, desc=f"Running batch inference on {self.device}"):
            batch = images[i : i + self.batch_size].to(self.device)
            outputs = self.model(batch, names_and_subjects=names_and_subjects)

            for dataset_name in outputs:
                for subject_id in outputs[dataset_name]:
                    outputs[dataset_name][subject_id] = outputs[dataset_name][subject_id].detach().cpu()
            results.append(outputs)

        #concatenate all batches
        final_results = {}
        for batch in results:
            for dataset_name in batch:
                if dataset_name not in final_results:
                    final_results[dataset_name] = {}
                for subject_id in batch[dataset_name]:
                    if subject_id not in final_results[dataset_name]:
                        final_results[dataset_name][subject_id] = []
                    final_results[dataset_name][subject_id].append(batch[dataset_name][subject_id])

        #concatenate the lists
        for dataset_name in final_results:
            for subject_id in final_results[dataset_name]:
                final_results[dataset_name][subject_id] = torch.cat(
                    final_results[dataset_name][subject_id], dim=0
                )

        return final_results
    

    def run_single_subject_inference(
        self,
        images: torch.Tensor,
    ) -> dict:
        results = []
        # Handles the last batch even if it's smaller than batch_size
        for i in tqdm(range(0, len(images), self.batch_size), total=len(images)//self.batch_size + 1, desc=f"Running batch inference on {self.device}"):
            batch = images[i : i + self.batch_size].to(self.device)
            outputs = self.model(batch)

            outputs = outputs.detach().cpu()
            results.append(outputs)

        final_results = torch.cat(results, dim=0)
        return final_results

    @torch.no_grad()
    def run(
        self, 
        images: list[Image.Image], 
        names_and_subjects: Union[dict, None] = {"NSD": "all"}
    ) -> Union[dict, torch.Tensor]:
        
        images = [imagenet_transforms(image) for image in images]
        images = torch.stack(images, dim=0)
        assert images.ndim == 4, f"Expected images to be a 4D tensor, but got a {images.ndim}D tensor instead"

        if self.is_single_subject_model:
            if names_and_subjects is not None:
                print("\033[93mModel is a single-subject model, but names_and_subjects were provided. Ignoring names_and_subjects. To avoid this warning, set names_and_subjects to None when calling .run()\033[0m")

            ## returns a torch tensor
            return self.run_single_subject_inference(
                images=images,
            )
        else:
            ## returns a dict
            return self.run_multi_subject_inference(
                images=images,
                names_and_subjects=names_and_subjects,
            )

        

    @torch.no_grad()
    def plot(
        self,
        image: Image.Image,
        save_as: str,
        dataset_name: str = "NSD",
        subject_id: int = 1,
        mode = "inflated",
    ):
        assert isinstance(subject_id, int), f"subject_id must be an integer, but got: {type(subject_id)}"
        assert dataset_name in list(num_subjects.keys()), f"Dataset name {dataset_name} is not valid. Please choose from {list(num_subjects.keys())}."
        assert mode in valid_plot_modes, f"mode must be one of {valid_plot_modes}, but got: {mode}"

        result = self.run(
            images=[image], names_and_subjects={dataset_name: [subject_id]}
        )

        if self.is_single_subject_model:
            assert torch.is_tensor(result), f"Expected result to be a torch tensor for single-subject models, but got: {type(result)}"
            voxel_activations = result
        else:
            voxel_activations = result[dataset_name][f"sub-{subject_id:02}"]

        if self.model.vertices == 'visual':
            rois = [f"GlasserGroup_{x}" for x in range(1, 6)]
        elif self.model.vertices == 'all':
            rois = [f"GlasserGroup_{x}" for x in range(1, 23)]
        else:
            raise ValueError(f"Model vertices attribute must be 'visual' or 'all', but got: {self.model.vertices}")
        
        all_voxels = SelectROIs(selected_rois=rois).sample2wb(voxel_activations.numpy().squeeze())

        stat = hcp.cortex_data(all_voxels)
        vmin = np.nanmin(stat)
        vmax = np.nanmax(stat)

        plotting_mode = getattr(hcp.mesh, mode)
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
        print(f"Saved: {save_as}")
        return html_thing