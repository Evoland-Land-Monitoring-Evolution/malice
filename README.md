# Multimodal Multiview Self-supervised learning for Satellite Image Time Series (SITS)

- This code will first propose a SSL framework where :
  - S1  and SITS are independtly encoded by an U-TAE
  - The VicRegL loss are implemented at the bottleneck of the U-TAE and at the output (local and global features)
## Installation
  In the conda env:  `pip install -e . `
## Pre-training
## Downstream task
### Change Detection
Used a change detection dataset built using data from RPG seq on PASTIS ROIs.
#### Comparison with baselines
- Presto
- ALISE
- Interpoled S2
- DTW S2
