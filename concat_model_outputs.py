#!/usr/bin/env python3

import os
import re
import pandas as pd
from glob import glob

# use overly-complicated way to regex the model outputs
files = []
for fname in os.listdir('.'):
    if re.match(r'psd_out\d+.csv', fname):
        files.append(fname)

mainfile_srch = glob('./psd_out_all.csv')
if len(mainfile_srch) == 1:
    mainfile = pd.read_csv(mainfile_srch[0])
else:
    mainfile = pd.DataFrame()

# concatenate all the psd_out[job number].csv files together
# and append to the main file, if it already exists, and overwrite it.
# else will just save all the concatenated outputs as the new main file
dfs = []
for fl in files:
    df = pd.read_csv(fl)
    dfs.append(df)

newfile = mainfile.append(dfs)

# quality check on dataframe before overwriting
test_index = pd.Index(['AUC', 'se', 'pvalue', 'perm_scores', 'rec_name'])

if newfile.columns.identical(test_index) and len(newfile) > len(mainfile):
    newfile.to_csv('./psd_out_all.csv', index=False)
else:
    print("Data to save has messed up columns or is no different than existing data. Data will not be overwritten.")
