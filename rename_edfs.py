#!/usr/bin/env python3

import os
import mne
import pandas as pd
import argparse


mne.utils.set_log_level('ERROR')
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
    # Change the filenames of the raw edf files
    # extract date and time from edf file, find the MRN for the patient name
    # from the edf filename in the excel table, and then rename and move.

    # make sure 0 wasn't specified for sheet
    if args.sheet == 0:
        raise ValueError("Don't specify 0 as first sheet--use 1 instead")

    flname = args.file
    sheet_idx = args.sheet - 1  # subtract 1 b/c first sheet is actually 0-index
    process_type = args.ptype
    cwd = args.cwd
    failed_rename = args.failed_rename
    save_noname = args.save_noname

    ptclass = pd.read_excel(flname, sheet_name=sheet_idx)
    # holy shit, the column names change ever so slightly whenever a new file is used.
    # try to find the relevant columns instead
    # I'm taking a gamble by taking the first match if there's more than one.
    mrn_col = ptclass.columns[ptclass.columns.str.contains('mrn', case=False)][0]
    name_col = ptclass.columns[ptclass.columns.str.contains('name', case=False)][0]

    # clean garbage unicode space characters from names
    ptclass[name_col] = ptclass[name_col].str.replace(u'\xa0', '')
    # also remove normal spaces
    ptclass[name_col] = ptclass[name_col].str.replace(' ', '')

    # get edf and text files from the pre-analysis directory
    edfiles = []
    for f in os.listdir(cwd):
        if f.lower().endswith(".edf") and not f.lower().startswith("._"):
            edfiles.append(f)

    txtfiles = []
    for f in os.listdir(cwd):
        if f.endswith(".txt") and not f.startswith("._"):
            txtfiles.append(f)

    if process_type not in ('control', 'patient'):
        raise ValueError("ptype must be either 'patient' or 'control'")

    if not os.path.exists(cwd + '/Converted/'):
        os.makedirs(cwd + '/Converted/')

    # now loop over all the edf files, make sure the corresponding text file matches,
    # then rename them both.
    for i in range(0, len(edfiles)):
        # get patient name (by taking the first element after splitting the filename where the '_' is)
        # and then turn the '~' into a ','

        # remove any spaces from the names made from the filenames too
        if failed_rename:
            pname_last = edfiles[i].split('_')[0]
            pname_first = edfiles[i].split('_')[1]
            pname = ','.join([pname_last, pname_first]).replace(' ', '')
        else:
            pname = edfiles[i].split('_')[0]
            pname = ','.join(pname.split('~')).replace(' ', '')

        # will select the first row where the name matches with a name in the excel file, and gives just the MRN
        # if the patient isn't in the file, print something that says so, and then skip this loop
        if process_type == "patient":
            try:
                pmrn = int(ptclass[mrn_col][ptclass[name_col].str.contains(pname, case=False, na=False)].iloc[0])
                dir_use = 'Converted/'
            # sometimes the mrn might just be missing and it'll try to convert a NaN to int which gives a
            # valueerror instead of an indexerror. oh my god.
            except (IndexError, ValueError):
                print("Patient " + pname + " is missing from the excel file!")
                if failed_rename or not save_noname:
                    # don't re-save if name is still missing
                    continue

                if not os.path.exists('Converted/No_MRN_Found/'):
                    os.makedirs('Converted/No_MRN_Found/')

                pname = '_'.join(pname.split(', '))
                pmrn = pname
                dir_use = 'Converted/No_MRN_Found/'
                # continue
        elif process_type == "control":
            pname = '_'.join(pname.split(','))
            pmrn = pname
            dir_use = 'Converted/'
        else:
            print("process_type must be either 'patient' or 'control'")
            break

        # get the date from the edf file
        try:
            # preload needs to be true cause some files have EDF+ annotations
            raw = mne.io.read_raw_edf(cwd + '/' + edfiles[i], preload=True)
        except ValueError as e:
            print(e)
            print("for file " + edfiles[i])
            continue

        # datetime of recording (as unix epoch)
        dtime = raw.info['meas_date']
        # convert to GMT
        # (since we've inadvertently been using GMT in the file names this whole time; might as well stick with it)
        # newtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime(dtime[0]))
        # ok so I *GUESS* datetime objects changed at some point??? So now I have to extract the date like this.
        newtime = dtime.strftime(format="%Y-%m-%d_%H-%M-%S")

        # create new file name
        newedfn = str(pmrn) + '_' + newtime + '.edf'
        newtxtfn = str(pmrn) + '_' + newtime + '.txt'

        idx = [j for j, s in enumerate(txtfiles) if str(edfiles[i][:-4]) in s]
        # then can select the text file that matches the edf file with this:
        # txtfiles[idx[0]]

        if not os.path.exists(dir_use + newedfn):
            print('Saving ' + newedfn)
            if len(idx) == 0:
                os.rename(edfiles[i], dir_use + newedfn)
            else:
                os.rename(edfiles[i], dir_use + newedfn)
                os.rename(txtfiles[idx[0]], dir_use + newtxtfn)

        else:
            print('File ' + newedfn + ' already exists, skipping..')


CLI.add_argument(
    "--cwd",
    type=str,
    default='.'
)

CLI.add_argument(
    "--file",
    type=str,
    default='/Volumes/groups/NICU/Clinical Research Coordinators/Studies & Screening/General Screening Log.xlsx'
)

CLI.add_argument(
    "--sheet",
    type=int,
    default=1
)

CLI.add_argument(
    "--ptype",
    type=str,
    default="patient"
)

CLI.add_argument(
    "--failed_rename",
    type=str2bool,
    nargs='?',
    const=True,
    default=False
)

CLI.add_argument(
    "--save_noname",
    type=str2bool,
    nargs='?',
    const=True,
    default=False
)


if __name__ == '__main__':
    args = CLI.parse_args()
    main(args=args)
