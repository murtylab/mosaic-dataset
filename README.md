<p align="center">
    <img src="https://github.com/murtylab/mosaic-dataset/raw/master/images/banner.png" alt="mosaic-dataset banner" width="50%">
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
    folder="./MOSAIC" 
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
    folder="./MOSAIC" 
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
    backbone_name="ResNet18",
    framework="multihead",
    subjects="all",
    vertices="visual"
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
        Image.open("cat.jpg").convert("RGB"),
        Image.open("cat.jpg").convert("RGB")
    ]
)
```

Visualizing model predictions

```python
inference.plot(
    image=Image.open("cat.jpg").convert("RGB"),
    save_as="predicted_voxel_responses.html",
    dataset_name="NSD",
    subject_id=1,
    mode="inflated",
)
```

Loading up stimulus info:

```python
stim_info = mosaic.get_stiminfo(
    dataset_name="deeprecon",
    folder="./MOSAIC"
)

print(stim_info.head())
```

Merging files for easier loading

```python
from mosaic.utils.merging import merge_hdf5_files
from mosaic.datasets import MergedDataset

merge_hdf5_files(
    files=[
        './MOSAIC/NSD/sub-01_NSD.hdf5',
        './MOSAIC/deep_recon/sub-01_deeprecon.hdf5'
    ],
    save_as="./merged-test.hdf5"
)

dataset = MergedDataset(
    filename="./merged-test.hdf5"
)

print(len(dataset))
```

Downloading resting state data

```python
from mosaic.datasets.resting_state import download_resting_state_data

download_resting_state_data(
    dataset="BMD", ## or "NSD" or "THINGS"
    subject=1,
    session=1,
    run=1,
    folder="./MOSAIC"
)
```

Downloading time series data

```python
from mosaic.datasets.timeseries import download_timeseries_data

download_timeseries_data(
    folder = "./data",
    dataset_name = "deeprecon"
)
```

Dev Setup

```bash
git clone git+https://github.com/Mayukhdeb/mosaic-dataset.git
cd mosaic-dataset
python setup.py develop
```
