import torch
from PIL import Image
from collections import defaultdict
import torchvision.transforms as transforms

imagenet_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


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
    def run(self, images: list[Image.Image], names_and_subjects: dict = {"NSD": "all"}):
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
