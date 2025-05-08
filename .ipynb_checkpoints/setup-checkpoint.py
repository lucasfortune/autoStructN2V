# setup.py
from setuptools import setup, find_packages

setup(
    name="autoStructN2V",
    version="0.1.0",
    author="Lucas Fortune",
    author_email="lucas.fortune@hhu.de",
    description="A two-stage denoising approach combining standard and structured Noise2Void",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/autoStructN2V",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "numpy>=1.19.0",
        "pillow>=8.0.0",
        "scikit-image>=0.18.0",
        "matplotlib>=3.3.0",
        "tqdm>=4.50.0",
        "tensorboard>=2.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b0",
            "flake8>=3.9.0",
        ],
    },
)