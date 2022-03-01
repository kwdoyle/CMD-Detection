#!/usr/bin/env python3

import os
import mne
import numpy as np
from glob import glob
from collections import Counter
import argparse
import matplotlib.pyplot as plt

import eeg_functions as eeg
import legacy_functions as old


plt.interactive(True)
mne.utils.set_log_level('ERROR')
CLI = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

"""Extract events for start/stop trying moving your R/L hand, save and raw.fif for experiment patient data"""

# TODO make separate functions that 1) find the events 2) save the plot 3) save the fif file
#  then can call the event creation and plotting functions from this in a new script that can be run
#  just to check if a file was recorded properly


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def plot_trigs(x, y):
    plt.figure().suptitle(y)
    plt.plot(x)
    plt.show()


def check_dc_chans(raw):
    # this is used if want to look at raw DC channel data
    trig_chan = mne.pick_channels(raw.info['ch_names'],
                                  include=['DC5', 'DC6', 'DC7', 'DC8'])
    chan = raw._data[trig_chan, :]
    # test plot the trigger channels
    plot_trigs(chan[0, :], y='DC5')
    plot_trigs(chan[1, :], y='DC6')
    plot_trigs(chan[2, :], y='DC7')
    plot_trigs(chan[3, :], y='DC8')


def process_triggers(raw, chan_dict, trig_thresh=2):
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

    # use this instead of setting outright to choose specific channels. (i.e., exclude DC7)
    _trig_hex = {
        'DC5': 0x8,
        'DC6': 0x10,
        'DC7': 0x40,
        'DC8': 0x80
    }

    # instead of forcing this to use DC5 through DC8 for each hex value,
    # just set each value in order of channels passed.
    # ...or set each channel individually? maybe.
    # pass a dict of the channels that map to what DC5 through DC8 should be?
    # this'll set them for each "main name" (i.e., DC5, DC6, DC7, DC8)
    trig_mult = np.array([_trig_hex[x] for x in list(chan_dict.keys())])
    # then choose which channels actually correspond to them.
    # I guess this should work fine, since the order of the keys and values match..

    # WOW, you need this ordered=True so that it DOESN'T order the indices obtained in accending order.
    # Normally this was fine, since the channels passed were in increasing order anyway.
    # but NOW if, for example, DC6 is DC1, it will put the index for DC1 first, messing up the index order of the data below.
    trig_idx = mne.pick_channels(raw.ch_names, list(chan_dict.values()), ordered=True)

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


# Turn the fix_dc7 component from "test_files.py" into its own function to use after ensuring an
# equal number of events between all event ids.
def fix_dc7(events):
    to_check = [(0, events[-1, 0] + 1)]
    for i_block, (block_start, block_end) in enumerate(to_check):
        print('============FIX DC7===============')
        print(f'Fixing DC7 in block {i_block} in range '
              f'[{block_start}, {block_end}]')
        is_start = True
        is_inst = True
        for i_event, t_event in enumerate(events):
            if block_end > t_event[0] > block_start:
                if is_start is True:  # This should be a start
                    if t_event[2] not in [10, 20, 50, 60]:
                        events[i_event, 2] -= 20

                # Now check the rest
                if is_inst is True and is_start is True:
                    # Start of instruction to start
                    is_inst = False  # Next one is not instruction
                    is_start = True  # Next one is a start
                elif is_inst is False and is_start is True:
                    # Start of block, next one is a stop instruction
                    is_inst = True
                    is_start = False
                elif is_inst is True and is_start is False:
                    # Start of instruction to stop, next one is stop
                    is_inst = False
                    is_start = False
                elif is_inst is False and is_start is False:
                    # Start of stop block, next one is a start instruction
                    is_inst = True
                    is_start = True

    return events


def check_id_pairs(events):
    count = Counter(events[:, 2])
    pairs = [(10, 20), (30, 40), (50, 60), (70, 80)]

    ids_have = list(count.keys())
    pairs_use = []
    for p in pairs:
        if p[0] in ids_have and p[1] in ids_have:
            pairs_use.append(p)

    id_check = any(eid in list(count.keys()) for eid in [10, 20, 50, 60])
    # don't use this num_chk--check same number per pair instead
    # num_chk = np.all(chk_arr == chk_arr[0]) # so if this was FALSE, then the below would go b/c "not false"
    num_count_chk = {}
    num_count = {}
    for pu in pairs_use:
        vals = (count[pu[0]], count[pu[1]])
        # stupid way to check if both numbers are the same
        chk = len(set(vals)) != 1
        num_count_chk[pu] = chk
        # also save unique counts
        num_count[pu] = np.unique(vals)

    # per other pair set, if counts are off by more than, e,g, 1
    # then do the below fix.
    pairs_alt_use = []
    count_thresh_chk = {}

    if (10, 20) in num_count.keys() and (30, 40) in num_count.keys():
        count1020 = num_count[(10, 20)]
        count3040 = num_count[(30, 40)]
        # TODO replace this "2" and the one below with an actual variable I can set
        if any(abs(count1020 - count3040) > 2):
            count_thresh_chk["(10,20),(30,40)"] = True
        else:
            count_thresh_chk["(10,20),(30,40)"] = False

    if (50, 60) in num_count.keys() and (70, 80) in num_count.keys():
        count5060 = num_count[(50, 60)]
        count7080 = num_count[(70, 80)]
        if any(abs(count5060 - count7080) > 2):
            count_thresh_chk["(50,60),(70,80)"] = True
        else:
            count_thresh_chk["(50,60),(70,80)"] = False

    # add checks for if the only pairs are (30,40) and (70,80)
    # !!! actually jusrt need to make sure there's only 1 unique number
    # for each pair instead of doing this
    # if (30, 40) in num_count.keys() and not (10, 20) in num_count.keys():

    return id_check, num_count, num_count_chk, count_thresh_chk


def clean_it(events):
    events2 = eeg.clean_events(events)
    events3 = eeg.clean_trigger_blocks2(events2)

    count = Counter(events3[:, 2])
    chk_arr = np.array(list(count.values()))
    if len(chk_arr) == 0:
        print('Could not clean events.')
        return events

    # also need to check if events have all the IDs before fixing dc7
    ids_ideal = [10, 20, 30, 40, 50, 60, 70, 80]
    ids_have = list(count.keys())
    # wow, this is awful that I have "ids_check" and "id_check" as 2 separate variables.
    ids_check = all(eid in ids_have for eid in ids_ideal)

    # id_pairs = check_id_pairs(events3)
    id_check, num_count, num_count_chk, count_thresh_chk = check_id_pairs(events3)

    # check if a single unique value for each id pair.
    # this ensures there's the correct number of events within 30,40 and 70,80
    # so that fix_dc7 will work correctly
    id_pair_chk = {}
    for key, value in num_count.items():
        if len(value) == 1:
            id_pair_chk[key] = True
        else:
            id_pair_chk[key] = False

    # manually set start/stop if all IDs have same number of events
    # and do NOT have all 8 IDs
    # if np.all(chk_arr == chk_arr[0]) and not ids_check:
    if all(list(id_pair_chk.values())) and not ids_check:
        print('DC7 failure--start/stop not defined. Manually setting instead')
        events3 = fix_dc7(events3)

    return events3


def process_file(raw, trig_thresh, chan_dict):
    raw = process_triggers(raw=raw, trig_thresh=trig_thresh, chan_dict=chan_dict)
    events = mne.find_events(raw, consecutive=True)
    # now clean them
    # do cleaning in single function
    # return an empty array if file has no triggers
    if events.size > 0:
        events = clean_it(events)
    else:
        return np.array([])

    return events


def main(wd, args):
    # make directories first
    if not os.path.exists(wd + '/event_plots/'):
        os.makedirs(wd + '/event_plots')

    if not os.path.exists(wd + '/fif_files/'):
        os.makedirs(wd + '/fif_files/')

    if not os.path.exists(wd + '/event_files/'):
        os.makedirs(wd + '/event_files/')


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
    chan_dict = args.chan_dict

    # process files
    for fl in files:
        print("Processing file: " + os.path.basename(fl))
        # First check if file was already processed. If so, skip it
        if not args.force_reprocess:
            if os.path.exists(wd + '/fif_files/' + '/' + os.path.basename(fl)[:-4] + '-raw.fif'):
                print('file already processed, skipping..')
                continue

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

        # un-comment if want to check dc channel data
        # check_dc_chans(raw)

        events = process_file(raw=raw, trig_thresh=0.5, chan_dict=chan_dict)

        if events.size > 0 and not args.force_old_method:
            id_check, num_count, num_count_chk, count_thresh_chk = check_id_pairs(events)
            # essentially the opposite of the check that determines if fix_dc7 should be used
            # if id_check and not num_chk:
            if (id_check and any(list(num_count_chk.values()))) or any(list(count_thresh_chk.values())):
                print('DC7 is unusable--re-running without it.')
                # remove DC7 from chan_dict if it exists (the None value prevents it from erroring if there is no DC7 key for whatever reason)
                chan_dict.pop('DC7', None)
                events = process_file(raw=raw, trig_thresh=0.5, chan_dict=chan_dict)

            # ensure all event IDs are present with equal number of events
            count = Counter(events[:, 2])
            chk_arr = np.array(list(count.values()))
            num_chk = np.all(chk_arr == chk_arr[0])
            id_check = any(eid in list(count.keys()) for eid in [10,20,30,40,50,60,70,80])

            if not num_chk and not id_check:
                print('Warning: still have unequal number of events or missing some event IDs.')

        # can probably combine this if statement with the above one.
        if not events.size > 0 or args.force_old_method:
            print('File has no events. Attempting to use old trigger processing method.')
            raw = old.process_older_recs(raw, fl=fl, wd=wd)
            events = mne.find_events(raw, consecutive=True)

        plt_name = os.path.basename(fl).split('/')[len(os.path.basename(fl).split('/')) - 1]
        # make blank plot if file has no events
        try:
            eventplt = mne.viz.plot_events(events, show=False)
        except ValueError:
            eventplt = plt.figure()

        eventplt.suptitle(plt_name)
        # save plots of events to see if preprocessing was successful
        eventplt.savefig(wd + '/event_plots/' + plt_name[:-4] + '.png')

        # Save fif files in their own directory for each conscious state group
        # create the fif files directory if it doesn't already exist
        raw.save(wd + '/fif_files/' + '/'+os.path.basename(fl)[:-4] + '-raw.fif', overwrite=True)
        # also write events, b/c this preprocessing step is too complicated to just have the cmd script perform as well.
        mne.write_events(wd + '/event_files/' + '/'+os.path.basename(fl)[:-4] + '-eve.fif', events)


CLI.add_argument(
    "--filetype",
    type=str,
    default='edf',
    help='specify if preprocessing edf or fif files'
)

CLI.add_argument(
    '--chan_dict',
    type=lambda e: {k:v for k,v in (x.split(':') for x in e.split(','))},
    default={'DC5': 'DC5', 'DC6': 'DC6', 'DC7': 'DC7', 'DC8': 'DC8'},
    help='comma-separated field:position pairs, e.g. Date:0,Amount:2,Payee:5,Memo:9'
)

CLI.add_argument(
    "--force_reprocess",
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
    help='choose not to skip over already-processed files'
)

CLI.add_argument(
    "--force_old_method",
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
    help='force the old eeg trigger processing method'
)


if __name__ == '__main__':
    wd = os.getcwd()
    args = CLI.parse_args()
    main(wd=wd, args=args)
