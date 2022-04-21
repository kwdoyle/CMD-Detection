#!/usr/bin/env python3

import os
import mne
import timeit
import warnings
import argparse
import traceback
from glob import glob
import pandas as pd

import eeg_functions as eeg

mne.utils.set_log_level('ERROR')
warnings.filterwarnings('ignore')
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
    cwd = args.cwd
    write_dir = args.write_dir
    events_dir = args.events_dir
    num_job = args.num_job
    combined_output_fl = args.combined_output_fl
    rawfiles = args.rawfiles
    nperm = args.nperm
    is_control = args.is_control

    os.chdir(cwd)

    # set parameters
    fmin = 1.0
    fmax = 30.0
    tmin = 0.0
    tmax = 2.0
    n_epo_segments = 5
    pipe_type = "psd"
    overlap = 0.9
    delays = [1, 2, 4, 8]
    bands = ((1, 3), (4, 7), (8, 13), (14, 30))
    n_permutations = nperm  # 500  # main analysis nejm v1
    # n_permutations = 2000  # for FDR at the rec level for review 1
    use_ch = [u'C3', u'C4', u'O1', u'O2', u'Cz', u'F3', u'F4', u'F7', u'F8', u'Fz',
              u'Fp1', u'Fp2', u'P3', u'P4', u'Pz', u'T7', u'T8', u'P7', u'P8']
    laplac_ref = True

    # This needs a try/except around it to prevent two jobs from seeing the dir doesn't exist
    # and then both trying to write the dir
    if not os.path.exists(write_dir):
        try:
            os.makedirs(write_dir)
        except FileExistsError:
            print("outfile directory already made")

    print("Analyzing files " + str(rawfiles), flush=True)

    for i in range(len(rawfiles)):
        start_time = timeit.default_timer()

        dat = rawfiles[i]
        print('Processing file ' + dat)

        flnm_chk = dat.split('/')[len(dat.split('/'))-1][:-8]
        event_fl = glob(events_dir + '/' + flnm_chk + '-eve.fif')
        # this should allow this script to analyze pre and post fede box recordings and will set the event file
        # if one exists
        if len(event_fl) > 0:
            event_fl = event_fl[0]
            read_func = eeg.read_data_fedebox
        else:
            read_func = eeg.read_data

        # Check if this file name has already been written to the output file. If so, can skip it.
        # That way you can run this script over an existing directory of files, of which you can add new ones to
        # without having to re-analyze the files that have already been analzyed.
        combined_output_flname = glob(write_dir + '/' + combined_output_fl)

        if len(combined_output_flname) > 0:
            combined_output_flname = combined_output_flname[0]

            combined_output = pd.read_csv(combined_output_flname)
            if combined_output['rec_name'].str.contains(flnm_chk).any():
                print('File ' + dat + ' has already been analyzed. Skipping...')
                continue

        else:
            print('Combined model output file does not exist or incorrectly specified. '
                  'Continuing without checking for previously analyzed files.')

        # ugh, just use pandas to save and read the output files instead of csv reader.
        #  That way I can more easily read by column b/c right now, the column names aren't defined.
        #  ...but I GUESS the recording names will always be in the 4th column...
        #  ..but maybe I should improve this anyway? but then combining the outputs will be more difficult..
        #  ..can't just cat them all into a new file..

        epochs = read_func(data=dat,
                           event_fl=event_fl,
                           use_ch=use_ch,
                           tmin=tmin,
                           tmax=tmax,
                           fmin=fmin,
                           fmax=fmax,
                           n_epo_segments=n_epo_segments,
                           hand_use=None,
                           is_control=is_control)

        if not epochs:
            print('\n')
            # save file with null data as well if no epochs
            psd_out = (0, 0, 0, 0, dat)
            tmp_df = pd.DataFrame([list(psd_out)], columns=['AUC', 'se', 'pvalue', 'perm_scores', 'rec_name'])
            filename = write_dir + '/' + 'psd_out' + num_job + '.csv'
            with open(filename, 'a') as f:
                tmp_df.to_csv(f, mode='a', index=False, header=not f.tell())

            continue

        # need to include the epoch-fixing in this script too. This pipeline also won't run if mne.Epochs rejects
        # some epochs on its own accord. Will get a 'class not balanced' error from run_pipeline otherwise.
        event_lens = [len(epochs[eid]) for eid in epochs.event_id]
        if not all(val == event_lens[0] for val in event_lens):
            print('Event lengths are not equal; fixing them..')
            epochs = eeg.fix_epochs(epochs, good_len=n_epo_segments * 2)

        # save event plots after cleaning them again before processing
#        eventplt = mne.viz.plot_events(epochs.events, show=False)
#        plt_name = os.path.basename(dat).split('/')[len(os.path.basename(dat).split('/')) - 1]
#
#        if not os.path.exists('./event_plots_for_pipeline/'):
#            os.makedirs('./event_plots_for_pipeline/')
#        eventplt.savefig('./event_plots_for_pipeline/' + plt_name[:-8] + '.png')
#        eventplt.clf()

        print('Running pipeline with ' + str(nperm) + ' permutations')
        try:
            psd_out = eeg.run_pipeline(epochs=epochs,
                                       fmin=fmin,
                                       fmax=fmax,
                                       pipe_type=pipe_type,
                                       overlap=overlap,
                                       delays=delays,
                                       bands=bands,
                                       n_permutations=n_permutations,
                                       laplac_ref=laplac_ref)

            psd_out = psd_out + (dat,)
            # to insert as columns, needs to be a list of a list..
            tmp_df = pd.DataFrame([list(psd_out)], columns=['AUC', 'se', 'pvalue', 'perm_scores', 'rec_name'])

        except:
            psd_out = (0, 0, 0, 0, dat)
            tmp_df = pd.DataFrame([list(psd_out)], columns=['AUC', 'se', 'pvalue', 'perm_scores', 'rec_name'])
            print(traceback.print_exc())
            print(rawfiles[i])

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        print('Total time elapsed: ' + str(elapsed))
        print('\n')

        # write to file. if file has already been written to, this will not write the header a second time
        filename = write_dir + '/' + 'psd_out' + num_job + '.csv'
        with open(filename, 'a') as f:
            tmp_df.to_csv(f, mode='a', index=False, header=not f.tell())


CLI.add_argument(
    "--cwd",
    type=str,
    default='.',
    help='the working directory to run this script from'
)

CLI.add_argument(
    "--write_dir",
    type=str,
    default='./model_outfiles/',
    help='directory to save the model output to. directory will be created if it does not exist'
)

CLI.add_argument(
    "--events_dir",
    type=str,
    default='./event_files/',
    help='directory containing the event files corresponding to each fif file'
)

CLI.add_argument(
    "--num_job",
    type=str,
    default='0',
    help='the current job number. this is automatically set when running this script via the cluster'
)

CLI.add_argument(
    "--rawfiles",
    nargs="*",
    type=str,
    default=None,
    help='the list of fif files to analyze. this is usually set using set_files.sh, '
         'but file paths can be also passed manually here'
)

CLI.add_argument(
    "--combined_output_fl",
    type=str,
    default='psd_out_all.csv',
    help='file that contains all the previous model output. '
         'recordings in this file that were previously analyzed will be skipped'
)

# this argument is mainly to reduce number of permutations when testing
CLI.add_argument(
    "--nperm",
    type=int,
    default=500,
    help='the number of permutations to run when checking the model AUC significance'
)

CLI.add_argument(
    "--is_control",
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
    help='specify if analyzing a control file or not. mainly due to controls only containing one hand. '
         'normally, a file is skipped if it does not contain both hands'
)


if __name__ == '__main__':
    args = CLI.parse_args()
    main(args)
