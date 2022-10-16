# IGLU: EffIcient GCN Training via Lazy Updates

This repository contains code accompanying the paper [IGLU: EffIcient GCN Training via Lazy Updates](https://openreview.net/forum?id=5kq11Tl1z4). 

The repository is subdivided into two key directories:

- ```src``` - This directory contains the main runner scripts, along with dataset specific architectures. Further details are presented within the directory.
- ```makedata``` - This directory contains instructions for creating data in the format IGLU uses. 

## Dependencies

- For OGB Datasets (Proteins and Products): Compatible version of Pytorch Geometric is needed. Installation instructions can be used from this [link](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

- For other datasets, we use standard python packages - NumPy, SciPy, Scikit-Learn, Json, NetworkX (Older Version 1.x might be required).
- Tensorflow Version Used: 1.15.2 


## Queries 

In case of any questions, feel free to raise an issue. 

## Citing our work 

To cite our work, kindly use the BibTeX below. 

```
@inproceedings{
narayanan2022iglu,
title={{IGLU}: Efficient {GCN} Training via Lazy Updates},
author={S Deepak Narayanan and Aditya Sinha and Prateek Jain and Purushottam Kar and Sundararajan Sellamanickam},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=5kq11Tl1z4}
}
```
