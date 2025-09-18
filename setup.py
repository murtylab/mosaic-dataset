import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mosaic-dataset",
    version="0.0.0",
    description="mosaic",
    author="Benjamin Lahner, Mayukh Deb, N. Apurva Ratan Murty, Aude Oliva",
    author_email="blahner@mit.edu; mayukh@gatech.edu; ratan@gatech.edu; oliva@mit.edu",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mayukhdeb/mosaic-dataset",
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