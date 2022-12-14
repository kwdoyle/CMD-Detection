#!/usr/bin/env python3

import sys
import parse_cmd as cmd
import pandas as pd
import argparse

CLI = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


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
    rm_file = args.rm_file
    remove_recs = args.remove_recs
    if remove_recs:
        # NOTE: the sheet index might change in the future.
        remove_file = pd.read_excel(rm_file, sheet_name=1, engine='openpyxl')
        # clean
        remove_file = remove_file[['MRN', 'aud_datetime']]
        remove_file = remove_file.dropna(subset=['MRN'])
        remove_file['MRN'] = remove_file['MRN'].astype(int)
        datetime_nm = remove_file.aud_datetime.astype(str)
        datetime_nm = datetime_nm.replace(' ', '_', regex=True)
        datetime_nm = datetime_nm.replace(':', '-', regex=True)
        newflnm = remove_file.MRN.astype(str) + '_' + datetime_nm
        newflnm = newflnm + '-raw.fif'
    else:
        newflnm = None

    newout, badrecs = cmd.read_file2(input_file=model_output_file, rm_names=newflnm, rm_mcsp_cs=rm_mcsp_cs)
    # find cmd status per patient
    cmd_status = newout.groupby('mrn').apply(lambda x: any(x.p_signif == True)).reset_index()
    cmd_status = cmd_status.rename(columns={0:"patient_cmd_status"})

    newout = newout.merge(cmd_status, on='mrn')

    savenm = model_output_file.split('/')[len(model_output_file.split('/')) - 1]
    savenm2 = savenm.split('.csv')[0] + '_w_cmd'
    newout.to_csv('./' + savenm2 + '.csv', index=False)


CLI.add_argument(
    "--rawfl",
    type=str,
    default='./psd_out_all_w_crsr_group.csv',
    help='file to calculate cmd status for'
)

CLI.add_argument(
    "--remove_recs",
    type=str2bool,
    nargs='?',
    const=True,
    default=True,
    help='set to True to remove files listed in the rm_file specified'
)

CLI.add_argument(
    "--rm_file",
    type=str,
    default='/mnt/babylon/NICU/Long-Term Recovery Project/Long-Term Recovery New Cohort Master List_12.17.2021.xlsx',
    help='list of recordings to remove'
)

CLI.add_argument(
    "--rm_mcsp_cs",
    type=str2bool,
    nargs='?',
    const=True,
    default=True,
    help='remove the mcsp and cs recordings before cmd calculation'
    )

if __name__ == '__main__':
    args = CLI.parse_args()
    main(args)
