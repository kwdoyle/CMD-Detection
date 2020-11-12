#!/usr/bin/env python3

import os
import mne
import numpy as np
from glob import glob
import argparse
import matplotlib.pyplot as plt


plt.interactive(True)
mne.utils.set_log_level('ERROR')
CLI = argparse.ArgumentParser()

"""Extract events for start/stop trying moving your R/L hand, save and raw.fif for experiment patient data"""


def plot_trigs(x, y):
    plt.figure().suptitle(y)
    plt.plot(x)
    plt.show()


def process_triggers(raw, trig_chans=None):
    _mcp_trig_map = {
        0xD8: 'inst_start/left',  # 216
        0xD0: 'start/left',  # 208
        0xC8: 'inst_start/right',  # 200
        0xC0: 'start/right',  # 192
        0x98: 'inst_stop/left',  # 152
        0x90: 'stop/left',  # 144
        0x88: 'inst_stop/right',  # 136
        0x80: 'stop/right'  # 128
    }

    _mcp_event_id = {
        'inst_start/left': 10,
        'start/left': 20,
        'inst_stop/left': 30,
        'stop/left': 40,

        'inst_start/right': 50,
        'start/right': 60,
        'inst_stop/right': 70,
        'stop/right': 80
    }

    if trig_chans is None:
        trig_chans = ['DC5', 'DC6', 'DC7', 'DC8']

    trig_mult = np.array([0x8, 0x10, 0x40, 0x80])
    trig_thresh = 2  # 2 Volts trigger

    trig_idx = mne.pick_channels(raw.ch_names, trig_chans)

    trig_data = (np.abs(
        raw._data[trig_idx, :]) > trig_thresh).astype(np.int)

    raw_trig_data = np.multiply(trig_data.T, trig_mult).T
    raw_trig_data = np.sum(raw_trig_data, axis=0)

    # Smooth out the triggers

    # triggers of less than this samples are considered as noise
    smooth_ms = 30
    smooth_samples = int(smooth_ms / 1000 * np.ceil(raw.info['sfreq']))
    smooth_trig_data = np.zeros_like(raw_trig_data)

    idx = 1
    prev_val = raw_trig_data[0]
    while idx < len(raw_trig_data) - 1:
        t_val = raw_trig_data[idx]
        if t_val != prev_val:
            # Warning: this will not work if trigger changes less than
            # smooth samples away from the end.
            t_val = raw_trig_data[idx + smooth_samples]
            smooth_trig_data[idx: idx + smooth_samples + 1] = t_val
            prev_val = t_val
            idx += smooth_samples
        else:
            smooth_trig_data[idx] = t_val
            idx += 1

    # Delete stairs

    # get all non_zero values
    non_zero_idx = np.where(smooth_trig_data != 0)[0]
    non_zero_vals = smooth_trig_data[non_zero_idx]

    # Find where this values augment
    diff_idx = np.where(np.diff(non_zero_vals) > 0)[0]
    diff_seq = non_zero_vals[diff_idx]

    # we will look for this sequence within a second of data
    find_seq = [0x8, 0x10, 0x40, 0x80]

    max_samples = int(raw.info['sfreq'])

    found_seq = []

    # look for the ocurrences of the sequence
    for i in range(len(diff_seq) - 3):
        if all(diff_seq[i:i + 4] == find_seq):
            # If the sequence match, get the IDX of the beggining and the end
            idx_end = non_zero_idx[diff_idx[i + 3]]

            # The start is the last time that value appears, get the first time
            idx_st = non_zero_idx[diff_idx[i]]

            st_val = smooth_trig_data[idx_st]
            while (smooth_trig_data[idx_st] == st_val):
                idx_st -= 1

            if idx_end - idx_st < max_samples:
                found_seq.append((idx_st + 1, idx_end))

    if len(found_seq) == 0:
        print('Warning: check sequence not found.')
    else:
        print('Found {} check sequences.'.format(len(found_seq)))

    for seq_st, seq_end in found_seq:
        smooth_trig_data[seq_st:seq_end + 1] = 0

    max_value = 0xD8
    next_val = 0xD0

    # Find where this values change
    diff_idx = np.where(np.diff(smooth_trig_data) != 0)[0]
    diff_seq = smooth_trig_data[diff_idx]

    max_idx = np.where(diff_seq == max_value)[0]
    for idx in max_idx:
        if idx == len(diff_seq) - 1 or diff_seq[idx + 1] != next_val:
            bad_idx_end = diff_idx[idx]
            bad_idx_st = bad_idx_end - 1
            while (smooth_trig_data[bad_idx_st] == max_value):
                bad_idx_st -= 1
            print('Found bad trigger at ({} - {})'.format(bad_idx_st, bad_idx_end))
            smooth_trig_data[bad_idx_st:bad_idx_end + 1] = 0

    filtered_trig_data = np.zeros_like(raw_trig_data)

    for key, value in _mcp_trig_map.items():
        t_idx = smooth_trig_data == key
        if np.sum(t_idx) == 0:
            continue
        filtered_trig_data[t_idx] = _mcp_event_id[value]

    # Delete isolated MAX VALUE Triggers

    # data_idx = mne.pick_types(raw.info, eeg=True)
    # Invert polarity
    # raw._data[data_idx] *= -1.
    if 'STI 014' in raw.ch_names:
        sti_idx = mne.pick_channels(raw.ch_names, ['STI 014'])
        raw._data[sti_idx] = filtered_trig_data
    else:
        new_raw = raw.copy().pick_channels([raw.ch_names[0]])
        new_raw._data[0] = filtered_trig_data
        new_raw.rename_channels({raw.ch_names[0]: 'STI 014'})
        new_raw.info['chs'][0]['kind'] = mne.channels.channels._human2fiff['stim']
        raw.add_channels([new_raw])

    return raw


def main(wd, args):
    filetype = args.filetype
    if filetype == 'edf':
        extension = '.edf'
    elif filetype == 'fif':
        extension = '.fif'
    else:
        raise ValueError("filetype must either be 'edf' or 'fif'")

    # get list of all files in directory script is run from
    files = glob(wd + '/' + '*' + extension)

    exclude = [u'LLC', u'RUC', u'CHIN1', u'CHIN2', u'EKG1', u'EKG2', u'LAT1', u'LAT2', u'RAT1', u'RAT2',
               u'CHEST', u'ABD', u'FLOW', u'SNORE', u'DIF5', u'DIF6', u'POS', u'DC2', u'DC3', u'DC4', u'DC5', u'DC6',
               u'DC7', u'DC8', u'DC9', u'DC10', u'OSAT', 'STI 014', u'Fpz', u'Event', u'LOC', u'ROC', u'LSTIM', u'RSTIM',
               u'F9', u'F10', u'T9', u'T10', u'P9', u'P10', u'A1', u'A2', ]

    # process files
    for fl in files:
        print("Processing file: " + os.path.basename(fl))
        # Might have .fif raw, unprocessed files if reading in de-identified files saved using mne.
        if filetype == 'edf':
            raw = mne.io.read_raw_edf(fl, preload=True, misc=exclude)
        elif filetype == 'fif':
            raw = mne.io.read_raw_fif(fl, preload=True)
        else:
            raise ValueError("Unrecognized filetype specified")

        # montage = mne.channels.read_montage('standard_1020')
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage, on_missing='ignore')
        sfreq = raw.info['sfreq']

        # new trigger processing function
        raw = process_triggers(raw=raw)

        trig_chan = mne.pick_channels(raw.info['ch_names'],
                                      include=['DC5', 'DC6', 'DC7', 'DC8'])

        # this is used if want to look at raw DC channel data
        chan = raw._data[trig_chan, :]
        # test plot the trigger channels
        # plot_trigs(chan[0, :], y='DC5')
        # plot_trigs(chan[1, :], y='DC6')
        # plot_trigs(chan[2, :], y='DC7')
        # plot_trigs(chan[3, :], y='DC8')

        events = mne.find_events(raw, consecutive=True)

        plt_name = os.path.basename(fl).split('/')[len(os.path.basename(fl).split('/')) - 1]
        # make blank plot if file has no events
        try:
            eventplt = mne.viz.plot_events(events, show=False)
        except ValueError:
            eventplt = plt.figure()

        eventplt.suptitle(plt_name)
        # save plots of events to see if preprocessing was successful
        if not os.path.exists(wd + '/event_plots/'):
            os.makedirs(wd + '/event_plots')

        eventplt.savefig(wd + '/event_plots/' + plt_name[:-4] + '.png')

        # Save fif files in their own directory for each conscious state group
        # create the fif files directory if it doesn't already exist
        if not os.path.exists(wd + '/fif_files/'):
            os.makedirs(wd + '/fif_files/')

        raw.save(wd + '/fif_files/' + '/'+os.path.basename(fl)[:-4] + '-raw.fif', overwrite=True)


CLI.add_argument(
    "--filetype",
    type=str,
    default='edf'
)


if __name__ == '__main__':
    wd = os.getcwd()
    args = CLI.parse_args()
    main(wd=wd, args=args)
