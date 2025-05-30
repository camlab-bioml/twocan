from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="twocan",
    version="0.1.2",
    packages=find_packages(exclude=["examples*", "notebooks*", "tests*", "docs*"]),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "opencv-python>=4.5.0",
        "scikit-image>=0.18.0",
        "scikit-learn>=1.0.0",
        "spatialdata>=0.1.0",
        "optuna>=3.0.0",
        "tifffile>=2021.1.1",
        "matplotlib>=3.3.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "isort>=5.0",
            "pre-commit>=2.0",
            "jupyter>=1.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme",
            "myst-parser",
            "myst-nb",
            "nbsphinx",
            "sphinx-autobuild",
            "sphinx-copybutton",
            "sphinxcontrib-bibtex",
        ],
        "notebooks": [
            "jupyter>=1.0",
            "ipywidgets>=7.0",
        ],
    },
    author="Caitlin F. Harrigan",
    author_email="caitlin.harrigan@mail.utoronto.ca",
    description="A Bayesian optimization framework for multimodal registration of highly multiplexed single-cell spatial proteomics data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/camlab-bioml/twocan",
    project_urls={
        "Documentation": "https://twocan.readthedocs.io/",
        "Source": "https://github.com/camlab-bioml/twocan"
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    keywords="spatial-proteomics highly-multiplexed-imaging cross-modality-registration bayesian-optimization",
    include_package_data=True,
    zip_safe=False,
)
