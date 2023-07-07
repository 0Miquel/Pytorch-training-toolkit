import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tofu",
    version="1.0.0",
    author="Miquel Romero Blanch",
    author_email="miquel.robla@gmail.com",
    description="TorchFusion, a pytorch training pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/0Miquel/Pytorch-training-pipeline",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)