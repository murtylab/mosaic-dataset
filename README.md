# mosaic-dataset
python module to load the mosaic dataset (Lahner et al.)

```bash
pip install git+https://github.com/Mayukhdeb/mosaic-dataset.git
```

```python
import mosaic

dataset = mosaic.load(
    names_and_subjects={
        "nsd": [2,3],
        "deep_recon": "all",
    },
    folder="/research/datasets/mosaic-dataset" 
)

print(dataset[0])
```

Visualization

```python
import mosaic
from mosaic.utils import visualize
from IPython.display import IFrame

dataset = mosaic.load(
    names_and_subjects={
        "bold_moments": [1],
    },
    folder="/research/datasets/mosaic-dataset" 
)

visualize(
    betas=dataset[0]["betas"],
    ## set rois to None if you want to visualize all of the rois
    rois=[
        "L_FFC",
        "R_FFC",
    ],
    ## other modes are: 'white', 'midthickness', 'pial', 'inflated', 'very_inflated', 'flat', 'sphere'
    mode = "midthickness",
    save_as = "plot.html",
)
```
Loading pre-trained models

```python
import mosaic

model = mosaic.from_pretrained(
    backbone_name="resnet18",
    vertices="visual",
    framework="multihead",
    subjects="all"
)
```

Running inference with pre-trained models:

```python
from mosaic.utils.inference import MosaicInference
from PIL import Image

inference = MosaicInference(
    model=model,
    batch_size=32,
    device="cuda:0"
)

results = inference.run(
    images = [
        Image.open("cat.jpg"),
        Image.open("cat.jpg")
    ]
)

## (2, num_voxels)
print(results.shape)
```