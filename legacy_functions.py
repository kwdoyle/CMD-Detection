import mne
import numpy as np
from glob import glob

import eeg_functions as eeg


def create_channel(raw, ch_name):
    stim_data = np.zeros((1, len(raw.times)))
    stim_info = mne.create_info([ch_name], raw.info['sfreq'], ['stim'])
    stim_raw = mne.io.RawArray(stim_data, stim_info)
    raw.add_channels([stim_raw], force_update_info=True)

    return raw


def process_older_recs(raw, fl, wd, start_left=1, stop_left=2, start_right=3, stop_right=4):
    # call filename 'fname1' for compatability reasons. I don't feel like changing
    # any instances of 'fname1' in this.
    fname1 = fl
    # get list of text files
    # txtfiles = []
    # for f in os.listdir(wd):
    #     if f.endswith(".txt"):
    #         txtfiles.append(f)
    txtfiles = glob(wd + '/' + '*.txt')

    index = [idx for idx, s in enumerate(txtfiles) if fname1[:-4] in s]
    # make sure the current edf has a text file
    if not index:
        print("file " + fname1 + " does not have a text file")
        return raw
    else:
        index = index[0]

    fname2 = txtfiles[index]

    sfreq = raw.info['sfreq']

    # take triggers from 'Event' channel, which will then be assigned a certain value depending on the type of event
    # and eventually set into the STI 014 channel.

    # first check if the file even has the trigger event channel
    if not any([ele for ele in ['Trigger Event', 'Event'] if (ele in raw.info['ch_names'])]):
        print('file ' + fname1 + ' is missing the trigger channel')
        return raw

    trig_chan = mne.pick_channels(raw.info['ch_names'],
                                  include=['Trigger Event'])  # this now needs to be ['Trigger Event']
    chan = raw._data[trig_chan, :]
    trigs = np.where(chan == 1)[1]
    # TODO ok, the transform_timestamps is broken AGAIN probably because of new text file issues.
    event_tab = eeg.transform_timestamps(fname2)
    labels = eeg.get_event_rows(event_tab, sfreq)

    # remove bad trigs
    # might not need this? If I allow a larger overlap window between event blocks, more get correctly assigned,
    # and then any misc. triggers will be instead removed by clean_trigger_blocks
    # --No, this step is still needed.
    chan, bad_trigs = eeg.remove_bad_trigs(chan,
                                           sec=25,
                                           smallsec=4,
                                           sfreq=sfreq,
                                           min_block_size=16)  # min_block_size=4 min_block_size=8

    # check if file has triggers at all
    if len(np.unique(chan)) == 1 and np.unique(chan)[0] == 0:
        print('file: ' + fname1 + ' has no triggers')
        print('\n')
        # notrigs.append(fname1)
        return raw

    # find thresh
    # does the dtime really just need to be increased from 4 to, let's say 6 seconds,
    # in order to fix some incorrectly assigned triggers at the start?
    thresh = eeg.find_thresh(labels,
                             dtime=20,
                             sfreq=sfreq,
                             dtime_add=6)
    # assign start trigs
    # !!! Can we instead just set this overlap time / dtime to something like 15 seconds
    # and then just not use remove_bad_trigs (but now use clean_trigger_blocks) to fix all these problems?
    chan = eeg.assign_start(trigs,
                            thresh,
                            chan,
                            start_right=start_right,
                            start_left=start_left)
    # assign stop
    # maybe need even larger time like 20 seconds of overlap?
    chan = eeg.assign_stop(trigs,
                           chan,
                           sfreq,
                           t_range=18,
                           start_right=start_right,
                           start_left=start_left,
                           stop_right=stop_right,
                           stop_left=stop_left)

    # if stop triggers couldn't be assigned using the time range of 16 s, try using a time range of 18 s instead.
    if (stop_left not in np.unique(chan) and start_left in np.unique(chan)) or \
            (stop_right not in np.unique(chan) and start_right in np.unique(chan)):
        chan = eeg.assign_stop(trigs, chan, sfreq, t_range=20)
        # if there STILL are no stop triggers, print something to let me know.
        if stop_left not in np.unique(chan) or stop_right not in np.unique(chan):
            print('stop triggers still not found for file: ' + fname1)

    # Run remove_bad_trigs again before maing new thresh.
    # can't just change the assign_start function to not re-assign "bad" trigs that were set to 0 to another val,
    # because I NEED it to do that in case it removed an actual good trigger
    # that happens to fall under the alice or tennis range.
    chan, bad_trigs = eeg.remove_bad_trigs(chan,
                                           sec=25,
                                           smallsec=4,
                                           sfreq=sfreq,
                                           min_block_size=16)

    # create new threshold for cleaning trigger blocks
    new_thresh = eeg.make_new_thresh(chan,
                                     sfreq,
                                     tbuff=10,
                                     start_right=start_right,
                                     start_left=start_left,
                                     stop_right=stop_right,
                                     stop_left=stop_left,
                                     sndpass=False)  # tbuff=5

    # check if the tbuff was too small for this file (i.e., too many thresholds were made)
    #        if len(new_thresh) > 6:
    #            new_thresh = eeg.make_new_thresh(chan, sfreq, tbuff=10, sndpass=False)
    # remove extra start/stop triggers
    # does this tbuff need to be smaller? Um.. ok, isn't actually used.
    chan = eeg.clean_trigger_blocks(chan,
                                    sfreq,
                                    new_thresh,
                                    start_right=start_right,
                                    start_left=start_left,
                                    stop_right=stop_right,
                                    stop_left=stop_left)

    # run clean bad trigs again
    chan, bad_trigs2 = eeg.remove_bad_trigs(chan,
                                            sec=25,
                                            smallsec=4,
                                            sfreq=sfreq,
                                            min_block_size=16)
    # min_block_size=4  # set back to 16 to remove the partial blocks that got clipped
    # (as well as any extra blocks that shouldn't be there)
    # so that the checker for those that do have misclipped events can work

    # get trigs again after removing bad ones
    new_trigs = np.where(chan != 0)[1]

    # now fix misassigned trigs
    # This function is actually CREATING mis-assigned triggers when the gap between
    # blocks is too small.
    # what if I don't use this one and only use that new function,
    # and make new_thresh w/ smaller tbuff?
    chan = eeg.clean_missassigned_trigs(chan,
                                        new_trigs,
                                        sfreq,
                                        start_right=start_right,
                                        start_left=start_left,
                                        stop_right=stop_right,
                                        stop_left=stop_left,
                                        btwn_block_time=20)  # 20

    # chan = eeg.clean_missassigned_trigs2(chan, new_thresh)

    # clean them again?
    # need to remake new_thresh again first in case a block was removed because it was too short
    # (this shouldn't be an issue once all are reclipped)
    # After extra start/stop triggers are removed, the tbuff doesn't seem to matter anymore?
    # MIGHT NEED TO PUT THIS BACK TO WHAT I HAD BEFORE
    new_thresh = eeg.make_new_thresh(chan,
                                     sfreq,
                                     tbuff=4,
                                     start_right=start_right,
                                     start_left=start_left,
                                     stop_right=stop_right,
                                     stop_left=stop_left,
                                     sndpass=True)

    chan = eeg.clean_missassigned_trigs2(chan,
                                         new_thresh,
                                         start_right=start_right,
                                         start_left=start_left,
                                         stop_right=stop_right,
                                         stop_left=stop_left)

    # and run this again for this one problem file which has another set of doubles that occur
    # at a slightly larger time before the other triggers, and are originally misassigned so they don't get fixed
    chan = eeg.clean_trigger_blocks(chan,
                                    sfreq,
                                    new_thresh,
                                    start_right=start_right,
                                    start_left=start_left,
                                    stop_right=stop_right,
                                    stop_left=stop_left)

    # update trigger channel with labeled events
    # coooool, some of the newer EDFs don't have an STI 014 channel.
    # gonna have to make it myself if it doesn't exist
    if 'STI 014' not in raw.ch_names:
        # raw.add_channels modifies raw in-place, so don't need to re-assign raw.
        create_channel(raw, 'STI 014')

    trig_chan = mne.pick_channels(raw.info['ch_names'], include=['STI 014'])
    raw._data[trig_chan, :] = chan

    return raw
