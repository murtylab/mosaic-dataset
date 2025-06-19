# mosaic-dataset
python module to load the mosaic dataset (Lahner et al.)

```bash
pip install git+https://github.com/Mayukhdeb/mosaic-dataset.git
```

BOLD5000

```python
from mosaic.datasets import BOLD5000SingleSubject

dataset = BOLD5000SingleSubject(
    folder="datasets/bold5000",
    subject_id=1, ## or 2
)
```

Generic Object Decoding

```python
from mosaic.datasets import (
    GenericObjectDecodingSingleSubject
)

dataset = GenericObjectDecodingSingleSubject(
    folder="data/god",
    subject_id=1,
)
```