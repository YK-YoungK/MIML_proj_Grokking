from setuptools import find_packages, setup

setup(
    name="grok",
    packages=find_packages(),
    version="0.0.1",
    install_requires=[
        "setuptools==59.5.0",
        "pytorch_lightning==1.5",
        "blobfile",
        "numpy",
        "torch==1.10",
        "tqdm",
        "scipy",
        "mod",
        "matplotlib",
        "sympy",
        "wandb==0.16.0",
    ],
)
