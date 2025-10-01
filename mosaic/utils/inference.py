import torch
from PIL import Image
import torchvision.transforms as transforms
import hcp_utils as hcp
import nilearn.plotting as plotting
from ..models.transforms import SelectROIs
from ..constants import num_subjects

imagenet_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)

valid_plot_modes = ['white', 'midthickness', 'pial', 'inflated', 'very_inflated', 'flat', 'sphere']

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

    @torch.no_grad()
    def run(self, images: list[Image.Image], names_and_subjects: dict = {"NSD": "all"}) -> dict:
        images = [imagenet_transforms(image) for image in images]
        images = torch.stack(images, dim=0)

        results = []

        # Handles the last batch even if it's smaller than batch_size
        for i in range(0, len(images), self.batch_size):
            batch = images[i : i + self.batch_size].to(self.device)
            outputs = self.model(batch, names_and_subjects=names_and_subjects)

            for dataset_name in outputs:
                for subject_id in outputs[dataset_name]:
                    outputs[dataset_name][subject_id] = outputs[dataset_name][subject_id].detach().cpu()
            results.append(outputs)
    
        return results

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
        voxel_activations = result[0][dataset_name][f"sub-{subject_id:02}"]

        rois = [f"GlasserGroup_{x}" for x in range(1, 6)]
        
        selected_roi_indices = SelectROIs(
            selected_rois=rois,
        ).selected_roi_indices

        parcel_map = hcp.mmp.map_all
        all_voxels = torch.zeros_like(torch.tensor(parcel_map), dtype=torch.float32)

        all_voxels[selected_roi_indices] = voxel_activations
        plotting_mode = getattr(hcp.mesh, mode)

        html_thing = plotting.view_surf(
            plotting_mode,
            surf_map=hcp.cortex_data(all_voxels.numpy()),
            threshold=0.0,
            bg_map=hcp.mesh.sulc
        )
        html_thing.save_as_html(save_as)
        print(f"Saved: {save_as}")
        return html_thing