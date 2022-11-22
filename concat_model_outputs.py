#!/usr/bin/env python3

import os
import re
import argparse
import pandas as pd
from glob import glob


CLI = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
CLI.add_argument(
    "--cwd",
    type=str,
    default='./',
    help='the working directory to run this script from'
)
CLI.add_argument(
    "--flsdir",
    type=str,
    default='./',
    help='the directory where the model outfiles to concat are'
)

args = CLI.parse_args()
wd = args.cwd
flsdir = args.flsdir

# regex the model outputs
files = []
for fname in os.listdir(flsdir):
    if re.match(r'psd_out\d+.csv', fname):
        files.append(fname)

mainfile_srch = glob(wd + '/psd_out_all.csv')
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
    newfile.to_csv(wd + '/psd_out_all.csv', index=False)
else:
    print("Data to save has messed up columns or is no different than existing data. Data will not be overwritten.")
