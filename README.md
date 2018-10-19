# Iterative Proportional Fitting (IPF)

From Wikipedia: IPF is an iterative algorithm for estimating cell values of a contingency table such that the marginal totals remain fixed and the estimated table decomposes into an outer product. Here we fit it using the method described on the Wikipedia page as well as a poisson regression.

## Basic scripts

The program consists of the following scripts
* ipf.py: contains the main functions and class for IPF
* main.py: runs the program on some toy data

## Notebooks

A notebook with an example is also included
* ipf_example.ipynb

## Example

```
import numpy as np
import pandas as pd
from ipf import ipf

x = np.array([[6,6,3],[8,10,10],[9,10,9],[3,14,8]])
row_total = np.array([20,30,35,15])
col_total = np.array([35,40,25])
ipf = ipf()
ipf.fit(initial_joint = x, marginal1 = row_total, marginal2 = col_total, method = 'mle')
```

## Authors

* **Syed Rahman**
