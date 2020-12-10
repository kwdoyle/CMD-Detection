#!/usr/bin/env python3

import os
import csv
import mne
import timeit
import warnings
import argparse
import traceback

import eeg_functions as eeg

# TODO have this script check the output script, if it exists, before analyzing a file and, if the file was already
#  analyzed and has output in the file, skip the current file and do not re-analyze it.
#  This way this script can be continuously run on a directory of files where new files can be continuously added
#  without the older files being reanalyzed each time.
mne.utils.set_log_level('ERROR')
warnings.filterwarnings('ignore')
CLI = argparse.ArgumentParser()


def main(args):
    cwd = args.cwd
    write_dir = args.write_dir
    num_job = args.num_job
    rawfiles = args.rawfiles
    nperm = args.nperm

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

        epochs = eeg.read_data2(data=dat,
                                use_ch=use_ch,
                                tmin=tmin,
                                tmax=tmax,
                                fmin=fmin,
                                fmax=fmax,
                                n_epo_segments=n_epo_segments,
                                hand_use=None)

        if not epochs:
            print('\n')
            continue

        # need to include the epoch-fixing in this script too. This pipeline also won't run if mne.Epochs rejects
        # some epochs on its own accord. Will get a 'class not balanced' error from run_pipeline otherwise.
        event_lens = [len(epochs[eid]) for eid in epochs.event_id]
        if not all(val == event_lens[0] for val in event_lens):
            print('Event lengths are not equal; fixing them..')
            epochs = eeg.fix_epochs(epochs, good_len=n_epo_segments * 2)

        # save event plots after cleaning them again before processing
        eventplt = mne.viz.plot_events(epochs.events, show=False)
        plt_name = os.path.basename(dat).split('/')[len(os.path.basename(dat).split('/')) - 1]

        if not os.path.exists(cwd + '/event_plots_for_pipeline/'):
            os.makedirs(cwd + '/event_plots_for_pipeline/')
        eventplt.savefig(cwd + '/event_plots_for_pipeline/' + plt_name[:-8] + '.png')

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

        except:
            psd_out = (0, 0, 0, 0, dat)
            print(traceback.print_exc())
            print(rawfiles[i])

        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        print('Total time elapsed: ' + str(elapsed))
        print('\n')

        # If the write_dir argument provided to the function has an end /, then this might not work
        with open(write_dir + '/' + 'psd_out' + num_job + '.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(psd_out)


CLI.add_argument(
    "--cwd",
    type=str,
    default='.'
)

CLI.add_argument(
    "--write_dir",
    type=str,
    default='./model_outfiles/'
)

CLI.add_argument(
    "--num_job",
    type=str,
    default='0'
)

CLI.add_argument(
    "--rawfiles",
    nargs="*",
    type=str,
    default=None
)

# this argument is mainly to reduce number of permutations when testing
CLI.add_argument(
    "--nperm",
    type=int,
    default=500
)


if __name__ == '__main__':
    args = CLI.parse_args()
    main(args)
