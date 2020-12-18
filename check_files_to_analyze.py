#!/usr/bin/env python3

import sys
from glob import glob
import pandas as pd
import numpy as np

"""
Use this script to determine which auditory recordings have not been analyzed.
Usage: check_files_to_analyze.py [model output file path] [fif file path]
"""

output = pd.read_csv(sys.argv[1])
fif_files = glob(sys.argv[2] + '*raw.fif')
fif_files_chk = [x.split('/')[len(x.split('/'))-1] for x in fif_files]

output_recs_raw = output['rec_name'].tolist()
output_recs = [x.split('/')[len(x.split('/'))-1] for x in output_recs_raw]

match_idx = np.where([x in output_recs for x in fif_files_chk])[0]
match_fls = np.array(fif_files_chk)[match_idx]
len(match_fls)

not_match_idx = np.where([x not in output_recs for x in fif_files_chk])[0]
not_match_fls = np.array(fif_files_chk)[not_match_idx]

# now get back full path of file
not_match_full_pths = np.array(fif_files)[not_match_idx]

np.savetxt("./files_to_analyze.txt", not_match_full_pths, newline=" ", fmt="%s")
