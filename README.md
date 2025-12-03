<p align="center">
  <img
    src="https://github.com/murtylab/mosaic-dataset/raw/master/images/banner.png"
    alt="mosaic-dataset banner"
    width="50%"
  >
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/murtylab/mosaic-dataset/blob/master/examples/mosaic-starter.ipynb">
    <img
      src="https://colab.research.google.com/assets/colab-badge.svg"
      alt="Open In Colab"
    >
  </a>
  <a href="https://github.com/murtylab/mosaic-dataset/actions/workflows/run_tests.yml">
    <img
      src="https://github.com/murtylab/mosaic-dataset/actions/workflows/run_tests.yml/badge.svg"
      alt="test mosaic"
    >
  </a>
</p>

Load the mosaic dataset ([Lahner et al.](https://www.biorxiv.org/content/10.64898/2025.11.28.690060v1)) and the associated pre-trained models

```bash
pip install mosaic-dataset
```

```python
import mosaic

dataset = mosaic.load(
    names_and_subjects={
        "NaturalScenesDataset": [1], ## set this to "all" if you want to download data for all subjects
        # "THINGS": "all"
    },
    folder="./MOSAIC",
    # if set to False, dataset[i]["betas"] will return a tensor containing the betas for all ROIs concatenated together
    parse_betas=True
)

print(dataset[0].keys())
```

Visualization

```python
import mosaic
from mosaic.utils import visualize

visualize(
    betas=dataset[0]["betas"],
    ## set rois to None if you want to visualize all of the rois
    rois=[
        "L_FFC",
        "R_FFC",
        "L_PHA2",
        "R_PHA2",
        "L_V1",
        "R_V1",

    ],
    ## other modes are: 'white', 'midthickness', 'pial', 'inflated', 'very_inflated', 'flat', 'sphere'
    mode = "inflated",
    save_as = "plot.html",
)
```
Loading pre-trained models

```python
import mosaic

model = mosaic.from_pretrained(
    backbone_name='ResNet18', 
    framework='multihead', 
    subjects='all', 
    vertices='visual', 
    folder='./mosaic_models/'
)

## or load a single subject model
single_subject_model = mosaic.from_pretrained(
    backbone_name="CNN8",
    framework="singlehead",
    subjects="sub-05_NSD",
    vertices="visual",
)
```

Running inference with pre-trained models:

```python
from mosaic.utils.inference import MosaicInference

inference = MosaicInference(
    model=model,
    batch_size=32,
    device="cpu"
)

results = inference.run(
    images = [
        Image.open("face.jpg").convert("RGB"),
    ],
    names_and_subjects={"NaturalScenesDataset": "all"}
)
```

Visualizing model predictions

```python
#note responses to the face are highest in the ventral stream
inference.plot(
    image=Image.open("face.jpg").convert("RGB"),
    save_as="predicted_voxel_responses.html",
    dataset_name="NaturalScenesDataset",
    subject_id=1,
    ## other modes are: 'white', 'midthickness', 'pial', 'inflated', 'very_inflated', 'flat', 'sphere'
    mode="inflated"
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
    folder = "./mosaic-timeseries",
    dataset_name = "deeprecon"
)
```

Dev Setup

```bash
git clone git+https://github.com/Mayukhdeb/mosaic-dataset.git
cd mosaic-dataset
python setup.py develop
```


## Running tests

**Note**: Before you start running tests, open [`tests/tesing_config.json`](https://github.com/murtylab/mosaic-dataset/blob/master/tests/testing_config.json) and make the following changes

1. Rename the `dataset_folder` and `models_folder` to wherever you already have the files or where you'd like to download them
2. set `download_pretrained_models` to `true` if you want to test the models *with* their actual trained weights. It's set to `false` by default to avoid spamming the AWS bucket with requests. 

```bash
pytest -vvx tests/
```

And to make the tests more verbose (with the prints displayed)

```bash
pytest -vvx --capture=no tests/
```
