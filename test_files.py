from pathlib import Path

import numpy as np
import mne
from collections import Counter

import matplotlib.pyplot as plt

fname = 'recording1-raw.fif'

db_path = Path('/Users/fraimondo/ownCloud/ICM/data/columbia/'
               'deidentified fifs 1')

"""
Elements for each subject's dict:
    'trig_tresh':           Threshold to use to consider one trigger as up, in
                            volts.
    'events_to_remove':     Samples where events should be removed
    'blocks_to_check':      List of tuples with the block start and end
                            samples. Only the ones in this list will be checked
    'keep_only_blocks':     If True, all the events that are not within the
                            defined blocks will be discarded
    'crop_events':          Tuple of first last event to consider. Can be
                            (None, X) or (X, None) to crop only on one side.
    'fix_dc7':              This asumes DC7 (start/stop) is bad and checks
                            that the events should happen in the right order
    'auto_check_block':     Automatically detect blocks and check them
    'events_to_inject':     List of lists with the events to inject.
"""


params_dict = {
    'recording1-raw.fif': {
        'trig_thresh': 0.5,
        'blocks_to_check': [
            (10000, 68000)
        ],
        'events_to_remove': [11414, 55124],
        'auto_check_block': True
    },
    'recording3-raw.fif': {
        'blocks_to_check': [
            (6500, 64000),
            (64000, 126000),
            (126000, 190000),
            (190000, 249000),
            (250000, 307000),
            (648000, 706107)],
        'keep_only_blocks': True,
        'fix_dc7': True
    },
    'recording4-raw.fif': {
        'crop_events': (4000, 374000),
        'fix_dc7': True,
        'auto_check_block': True
    },
    'recording5-raw.fif': {
        'trig_thresh': 0.5,
        'crop_events': (None, 368000),
        'blocks_to_check': [
            (0, 60000),
            (240000, 302000)
        ],
        'events_to_remove': [14748, 274950],
        'auto_check_block': True
    },
    'recording6-raw.fif': {
        'crop_events': (84450, 484000),
        'fix_dc7': True,
        'auto_check_block': True
    },
    'recording13-raw.fif': {
        'blocks_to_check': [
            (0, 62000),
            (65000, 130000),
            (130000, 190000),
            (190000, 251000),
            (251000, 316000),
            (316000, 380000)
        ],
        'events_to_inject': [
            [83947, 0, 30]
        ],
        'fix_dc7': True,
        'auto_check_block': True
    }
}


subject_dict = params_dict.get(fname, {})

raw = mne.io.read_raw_fif(db_path / fname, preload=True)

_mcp_trig_map = {
    0xD8: 'inst_start/left',    # 216 = 1101 1000b
    0xD0: 'start/left',         # 208 = 1101 0000b
    0xC8: 'inst_start/right',   # 200 = 1100 1000b
    0xC0: 'start/right',        # 192 = 1100 0000b
    0x98: 'inst_stop/left',     # 152 = 1001 1000b
    0x90: 'stop/left',          # 144 = 1001 0000b
    0x88: 'inst_stop/right',    # 136 = 1000 1000b
    0x80: 'stop/right'          # 128 = 1000 0000b
}

_mcp_event_id = {
    'inst_start/left': 10,      # 216 = 1101 1000b
    'start/left': 20,           # 208 = 1101 0000b
    'inst_stop/left': 30,       # 152 = 1001 1000b
    'stop/left': 40,            # 144 = 1001 0000b

    'inst_start/right': 50,     # 200 = 1100 1000b
    'start/right': 60,          # 192 = 1100 0000b
    'inst_stop/right': 70,      # 136 = 1000 1000b
    'stop/right': 80            # 128 = 1000 0000b
}

trig_chans = ['DC5', 'DC6', 'DC7', 'DC8']

trig_mult = np.array([0x8, 0x10, 0x40, 0x80])
trig_thresh = subject_dict.get('trig_thresh', 2)  # 2 Volts trigger

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

# This is to debug
idx = mne.pick_channels(raw.ch_names, include=trig_chans)

dc5 = raw.get_data()[idx[0]]
dc6 = raw.get_data()[idx[1]]
dc7 = raw.get_data()[idx[2]]
dc8 = raw.get_data()[idx[3]]
plt.figure()
plt.plot(dc5)
plt.title('dc5')
plt.figure()
plt.plot(dc6)
plt.title('dc6')
plt.figure()
plt.plot(dc7)
plt.title('dc7')
plt.figure()
plt.plot(dc8)
plt.title('dc8')

# End debug code

events = mne.find_events(raw, consecutive=True)

crop_events = subject_dict.get('crop_events', None)

if crop_events is not None:
    start = 0 if crop_events[0] is None else crop_events[0]
    end = events[-1, 0] + 1 if crop_events[1] is None else crop_events[1]
    crop_mask = np.logical_and(events[:, 0] > start, events[:, 0] < end)
    events = events[crop_mask]

keep_only_blocks = subject_dict.get('keep_only_blocks', False)

if keep_only_blocks is True:
    print('Keeping events only in the specified blocks')
    good_idx = np.zeros_like(events[:, 0]).astype(np.bool)
    to_check = subject_dict.get('blocks_to_check', [])
    for block_start, block_end in to_check:
        block_mask = np.logical_and(events[:, 0] > block_start,
                                    events[:, 0] < block_end)
        good_idx = np.logical_or(good_idx, block_mask)
    events = events[good_idx]

events_to_remove = subject_dict.get('events_to_remove', None)

if events_to_remove is not None:
    print(f'Marking {events_to_remove} as bad events')
    events = events[~np.in1d(events[:, 0], events_to_remove)]

events_to_inject = subject_dict.get('events_to_inject', None)
if events_to_inject is not None:
    events = np.r_[events, events_to_inject]
    events = events[events[:, 0].argsort()]

fig = mne.viz.plot_events(events)
fig.suptitle('Before fixes')


def _check_events(events, to_check):
    for i_block, (block_start, block_end) in enumerate(to_check):
        print('===========BLOCK CHECK==============')
        print(f'Checking block {i_block} in range '
              f'[{block_start}, {block_end}]')
        block_mask = np.logical_and(events[:, 0] > block_start,
                                    events[:, 0] < block_end)
        block_events = events[block_mask]

        # Events of same kind must be between 27.4 and 29.4 seconds appart

        for event_to_fix, event_id in _mcp_event_id.items():
            id_mask = block_events[:, 2] == event_id
            t_events = block_events[id_mask]
            diffs = np.diff(t_events[:, 0]) / raw.info['sfreq']
            bad_idx = np.where(np.logical_or(diffs < 27, diffs > 30))[0]
            if len(bad_idx):
                print(f'Found bad {event_to_fix} events diff: ')
                for t_idx in bad_idx:
                    print(f'\tEvent {t_idx + 1} @ {t_events[t_idx][0]} => '
                          f'diff is {diffs[t_idx]}')
                    should_remove = t_idx + 1 not in bad_idx
                    if should_remove:
                        print(
                            f'\tShould remove {t_idx} @ {t_events[t_idx][0]}')
        # Check that between the instruction trigger and the next we have 2.7s

        t_start = 0
        prev_inst = False
        for i_event, event in enumerate(events):
            if event[2] in [10, 30, 50, 70]:  # Inst, we count from here
                t_start = event[0]
                prev_inst = True
            else:
                diff = (event[0] - t_start) / raw.info['sfreq']
                if 2.6 > diff or diff > 2.8:
                    print(f'Event {i_event} @ {event[0]} does not have the '
                          f'preceding instruction in range (diff = {diff})')
                if prev_inst is False:
                    prev_start = event[0] - round(2.71875 * raw.info['sfreq'])
                    prev_code = event[2] - 10
                    print(f'\t Should inject [{prev_start}, 0, {prev_code}]')
                prev_inst = False

        event_counts = Counter(block_events[:, 2])
        print('This block event counts:')
        for t_name, t_id in _mcp_event_id.items():
            print(f'\t{t_name} => {event_counts[t_id]}')
        print('=====================================\n')
    event_counts = Counter(events[:, 2])
    print('Overall Event counts:')
    for t_name, t_id in _mcp_event_id.items():
        print(f'\t{t_name} => {event_counts[t_id]}')


to_check = subject_dict.get('blocks_to_check', [])
_check_events(events, to_check)

fix_dc7 = subject_dict.get('fix_dc7', False)
if fix_dc7:
    to_check = subject_dict.get('blocks_to_check', None)
    if to_check is None:
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

    fig = mne.viz.plot_events(events)
    fig.suptitle('Fixed DC7')


auto_check_block = subject_dict.get('auto_check_block', False)
if auto_check_block:
    n_events = events.shape[0]
    block_starts = events[slice(0, n_events, 32)][:, 0] - 1
    block_ends = events[slice(31, n_events + 1, 32)][:, 0] + 1
    _check_events(events, zip(block_starts, block_ends))


out_fname = db_path / fname.replace('.fif', '-eve.fif')
print(f'Saving events to {out_fname}')
mne.write_events(out_fname, events)
