# Paired IMC + IF slides from Harrigan et al. (Twocan-954 dataset)

Bayesian Optimization for Multimodal Registration of Highly Multiplexed Single-Cell Spatial Proteomics Data

* Paper 
* Data https://zenodo.org/records/15115811 (Two QC-pass samples are provided: `cell-line-0028-bd18455` and `tissue-0009-9073de4`)


## Run the example

```
conda activate twocan
snakemake -j 12 --use-conda 
```