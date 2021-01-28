#!/usr/bin/env python3

import sys
import parse_cmd as cmd
import pandas as pd


# '/Volumes/NeurocriticalCare/EEGData/Auditory/Consciousness/model_outfiles/psd_out_all_2020-12-28.csv'
if __name__ == '__main__':
    model_output_file = sys.argv[1]
    # model_output_file = '/Volumes/NeurocriticalCare/EEGData/Auditory/RECONFIG/Healthy Volunteers/all/Converted/model_outfiles/psd_out0.csv'
    # cmd_recs, noncmd_recs, bad_recs, perm_aucs = cmd.main(fl=model_output_file)
    newout, badrecs = cmd.read_file2(input_file=model_output_file)

    savenm = model_output_file.split('/')[len(model_output_file.split('/'))-1]
    savenm2 = savenm.split('.csv')[0] + '_w_cmd'
    # newout.to_csv('./psd_out_all_w_cmd.csv', index=False)
    newout.to_csv('./' + savenm2 + '.csv', index=False)
