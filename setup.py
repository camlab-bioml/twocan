from setuptools import setup, find_packages

setup(
    name="twocan",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "scikit-image",
        "scikit-learn",
        "spatialdata",
        "optuna",
        "tifffile",
    ],
    author="Caitlin F. Harrigan",
    author_email="caitlin.harrigan@mail.utoronto.ca",
    description="A Bayesian optimization framework for multimodal registration of highly multiplexed single-cell spatial proteomics data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/camlab-bioml/twocan",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
)
