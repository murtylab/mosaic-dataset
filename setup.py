import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mosaic",
    version="0.0.0",
    description="mosaic",
    author="Benjamin Lahner, Mayukh Deb, N. Apurva Ratan Murty, Aude Oliva",
    author_email="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mayukhdeb/mosaic-dataset",
    packages=setuptools.find_packages(),
    install_requires=None,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)