#!/usr/bin/env python3

import sys
import parse_cmd as cmd
import pandas as pd
import argparse

CLI = argparse.ArgumentParser()


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    model_output_file = args.rawfl
    rm_mcsp_cs = args.rm_mcsp_cs

    newout, badrecs = cmd.read_file2(input_file=model_output_file, rm_mcsp_cs=rm_mcsp_cs)
    # find cmd status per patient
    cmd_status = newout.groupby('mrn').apply(lambda x: any(x.p_signif == True)).reset_index()
    cmd_status = cmd_status.rename(columns={0:"patient_cmd_status"})

    newout = newout.merge(cmd_status, on='mrn')

    savenm = model_output_file.split('/')[len(model_output_file.split('/')) - 1]
    savenm2 = savenm.split('.csv')[0] + '_w_cmd'
    # newout.to_csv('./psd_out_all_w_cmd.csv', index=False)
    newout.to_csv('./' + savenm2 + '.csv', index=False)


CLI.add_argument(
    "--rawfl",
    type=str,
    default='/Volumes/NeurocriticalCare/EEGData/Auditory/cmd_outfiles/psd_out_all.csv'
)

CLI.add_argument(
        "--rm_mcsp_cs",
        type=str2bool,
        nargs='?',
        const=True,
        default=True
    )

if __name__ == '__main__':
    args = CLI.parse_args()
    main(args)
