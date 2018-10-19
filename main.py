import numpy as np
import pandas as pd
from ipf import ipf

def main():
    x = np.array([[6,6,3],[8,10,10],[9,10,9],[3,14,8]])
    row_total = np.array([20,30,35,15])
    col_total = np.array([35,40,25])
    ipf = ipf()
    print(ipf.fit(initial_joint = x, marginal1 = row_total, marginal2 = col_total, method = 'mle'))

if __name__ == '__main__':
    main()
