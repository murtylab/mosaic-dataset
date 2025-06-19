# mosaic-dataset
python module to load the mosaic dataset (Lahner et al.)

```bash
pip install git+https://github.com/Mayukhdeb/mosaic-dataset.git
```

```python
import mosaic

dataset = mosaic.load(
    # "bold5000", "generic_object_decoding", "nsd", "things_fmri", "deep_recon"
    name="bold5000", 
    subject_id="all", # or some integer (1-indexed)
)

print(dataset[0])
```