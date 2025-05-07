from os.path import exists, dirname, realpath
from setuptools import setup, find_packages
import sys

maintainer = "Raghava Alajangi"
maintainer_email = "raghava.alajangi95@gmail.com"
description = "Semantic Segmentation package for the LC project"
name = "CytoSegment-PyTorch"
year = "2022"

sys.path.insert(0, realpath(dirname(__file__)) + "/" + name)

setup(
    name=name,
    maintainer=maintainer,
    maintainer_email=maintainer_email,
    url="https://github.com/RaghavaAlajangi/CytoSegment-PyTorch.git",
    version="0.0.1",
    packages=find_packages(),
    package_dir={name: name},
    include_package_data=True,
    license="",
    description=description,
    long_description=open("README.md").read() if exists("README.md") else "",
    install_requires=[
        "click",
        "numpy",
        "matplotlib",
        "onnx",
        "pyyaml",
        "scikit-image",
        "tensorboard",
        "torch>=1.13.1",
        "torchvision>=0.13.1"
    ],
    python_requires=">=3.9, <4",
    keywords=["RT-DC", "segmentation"],
    classifiers=["Operating System :: OS Independent",
                 "Programming Language :: Python :: 3",
                 "Topic :: Scientific/Engineering",
                 "Intended Audience :: Private",
                 ],
    platforms=["ALL"],
)
