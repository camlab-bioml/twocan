# Paired FISH + IMC slides from Schultz et al. 

Simultaneous Multiplexed Imaging of mRNA and Proteins with Subcellular Resolution in Breast Cancer Tissue Samples by Mass Cytometry 

* Paper https://pubmed.ncbi.nlm.nih.gov/29289569/
* Data https://data.mendeley.com/datasets/m4b97v7myb/1  (only data from `Fig_2 FISH IMC comparison.7z` is used)


To re-run count RNA FISH spot counting you will need to additionally install [Polaris](https://doi.org/10.1016/j.cels.2024.04.006) before running `detect_spots.py`

## Run the example

```
conda activate twocan
python register.py

# uncomment this to re-run spot counting
# python detect_spots.py

python stat_fish.py
```