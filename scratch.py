import numpy as np
from pathlib import Path
import os
import csv
import pandas as pd

DATA_DIR = Path(r'C:\Users\QT3\Documents\EDUAFM Data')
# for f in p.glob('*.csv').iterdir():
#     print(f)
# get_string = 'C:\Users\QT3\Documents\EDUAFM Data
# # string = np.loadtxt(get_string, dtype=str, delimiter='_')
# # print(string)

with open(r'C:\Users\QT3\Documents\EDUAFM Data\Sample_ConstF_noStrainG_250px_100pps.csv', 'r') as file, open('test.csv', 'r') as test:
    dataframe = pd.read_csv(file, delimiter=';', header=None)
    test_file = pd.read_csv(test, delimiter=';', header=None)
    print(dataframe, test_file)
    dataframe.dropna(axis=1, how='all', inplace=True)
    test_file.dropna(axis=1, how='all', inplace=True)
    print(dataframe, test_file)
    print("row 0, col 250 :", dataframe.iat[0, 250])
    print("row 0, col 2 :", test_file.iat[0, 2])
