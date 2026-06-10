## CellGRN 


### Introduction

CellGRN is a framework that shifts the paradigm from global fitting to cell-specific reconstruction. CellGRN adopts a two-step strategy: it first leverages a global regulatory backbone from a global modeling strategy, and subsequently reconstructs cell-specific regulatory weights by projecting individual cell states onto this backbone.

CellGRN-sparse is an accelerated version of CellGRN, which skips the step of cell-specific GRN output and is optimized with sparse matrix and block computation.

All datasets can be downloaded from [Figshare](https://doi.org/10.6084/m9.figshare.31304758) or [Baidu Netdisk](https://pan.baidu.com/s/1mkpBOzyyuOHvmTPC4sMGBg?pwd=4r8v).

### Overview of CellGRN and CellGRN-sparse
----------------------------
![Workflow](imgs/demo.png)


### Package dependencies
- Python >=3.8, scanpy, scikit-learn,pyranges,umap-learn, anndata,pandas, numpy, scipy, jupyter 
- Visualization requires extra matplotlib, seaborn and R>=3.5.


### Installation

If you have not installed the Python dependencies above, we recommend using conda to install CellGRN in an independent environment. The conda tool (miniconda) can be installed from the [Anaconda website](https://docs.anaconda.com/miniconda/miniconda-install/).</br>

1. Clone this repository.

```Bash
git clone https://github.com/DELTA-TJ-submission/CellGRN
cd CellGRN
```

2. Create and activate a conda environment.

```Bash
conda env create -f cellgrn_env.yml
conda activate cellgrn
```

3. Install the local CellGRN package.

```Bash
pip install .
```

4. Test the installation in python

```python
import cellgrn
```

The installation typically takes a few minutes.

### Input and output format

The GRN backbone input should be a CSV file with three required columns:

| group_subtype | feature1 | feature2 |
| --- | --- | --- |
| TF-Gene | NFKB1 | IL6 |
| TF-Peak | NFKB1 | chr1:1000-1500 |
| Peak-Gene | chr1:1000-1500 | IL6 |

For `TF-Gene` rows, `feature1` is a TF and `feature2` is a gene. For `TF-Peak` rows, `feature1` is a TF and `feature2` is a peak. For `Peak-Gene` rows, `feature1` is a peak and `feature2` is a gene.

A example output (TF-Gene result) is as follows:

| TF | Gene | Score |
| --- | --- | --- |
| NFKB1 | IL6 | 100 |
| NFKB1 | IL8 | 99 |


### Tutorial
Two complementary examples are provided:

1. [tutorial.ipynb](./tutorial.ipynb): a compact tutorial for CellGRN and CellGRN-sparse using a 500-cell multiome dataset.
2. [code/melanoma_run.ipynb](code/melanoma_run.ipynb): a step-by-step CellGRN example on the melanoma cell line dataset used in the manuscript.



### Repository structure
- [code/grn_method](code/grn_method/): SCENIC+ and LINGER GRN inference script.
- [code/preprocess](code/preprocess/): Preprocessing scripts for input datasets and validation datasets.
- [code/***_run.ipynb](code/): Reproducible scripts to run CellGRN in multiome/spatial-multiome datasets in our manuscript. 
    - [code/scalability_run.ipynb](code/scalability_run.ipynb): Code for running time and memory assessment for different dataset sizes for CellGRN and CellGRN-sparse.
- [eval/***_eval.ipynb](eval/): Benchmark pipelines for each dataset.
- [eval/visualization/](eval/visualization/): R  scripts to plot figure in our manuscript.


## Reference
Cell-specific reconstruction improves gene regulatory network inference by resolving regulatory heterogeneity from single-cell multi-omics. (submitted)
