<p align="center">
    <img src="images/banner.png" alt="mosaic-dataset banner" width="50%">
</p>

<p align="center">
    <a href="https://colab.research.google.com/github/murtylab/mosaic-dataset/blob/master/examples/mosaic-starter.ipynb">
        <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
</p>

Load the mosaic dataset (Lahner et al.) and the associated pre-trained models

```bash
pip install mosaic-dataset
```

```python
import mosaic

dataset = mosaic.load(
    names_and_subjects={
        "NSD": [2,3],
        "deeprecon": "all",
    },
    folder="./mosaic_dataset" 
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
    folder="./mosaic_dataset" 
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
```

Visualizing model predictions

```python
inference.plot(
    image=Image.open("cat.jpg"),
    save_as="predicted_voxel_responses.html",
    dataset_name="NSD",
    subject_id=1,
    mode="inflated",
)
```

Dev Setup

```bash
git clone git+https://github.com/Mayukhdeb/mosaic-dataset.git
cd mosaic-dataset
python setup.py develop
```
