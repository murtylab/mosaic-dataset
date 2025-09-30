import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

"""
release checklist:
1. update version on `setup.py`
4. commit changes and push
5. make release on PyPI. Run the following commands:
    5.1 `python3 setup.py sdist bdist_wheel`
    5.2 (optional) `python3 -m pip install --user --upgrade twine`
    5.3 `python3 -m twine upload dist/*`
6. make a new release on github with the latest version
"""


setuptools.setup(
    name="mosaic-dataset",
    version="0.0.0",
    description="A scalable framework for fMRI dataset aggregation and modeling of human vision",
    author="Benjamin Lahner, Mayukh Deb, N. Apurva Ratan Murty, Aude Oliva",
    author_email="blahner@mit.edu; mayukh@gatech.edu; ratan@gatech.edu; oliva@mit.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/murtylab/mosaic-dataset",
    packages=setuptools.find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
