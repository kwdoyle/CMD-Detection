import re
import sys
import mne
import copy
import operator
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime
from collections import Counter

from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.tangentspace import TangentSpace
from pyriemann.estimation import Covariances
from pyriemann.estimation import CospCovariances

from pandas import DataFrame
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression

from pyriemann.estimation import HankelCovariances
# for PSD
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from mne.time_frequency import psd_multitaper
import pycsd


class CospBoostingClassifier(BaseEstimator, TransformerMixin):
    """Cospectral matrice bagging."""

    def __init__(self, baseclf):
        """Init."""
        self.baseclf = baseclf

    def fit(self, X, y):
        self.clfs_ = []
        for i in range(X.shape[-1]):
            clf = deepcopy(self.baseclf)
            self.clfs_.append(clf.fit(X[:, :, :, i], y))
        return self

    def predict_proba(self, X):
        proba = []
        for i in range(X.shape[-1]):
            proba.append(self.clfs_[i].predict_proba(X[:, :, :, i]))
        proba = np.mean(proba, axis=0)
        return proba

    def transform(self, X):
        proba = []
        for i in range(X.shape[-1]):
            proba.append(self.clfs_[i].predict_proba(X[:, :, :, i]))
        proba = np.concatenate(proba, 1)
        return proba
    

# ==============================================================================
# ## Function to take timestamp data from text file and convert times to seconds from start of experiment
# ==============================================================================

def transform_timestamps(data):
    with open(data) as f:
        lines = f.readlines()

    # strip out spaces, newlines, etc. from lines
    lines = [x.strip() for x in lines]

    # UPDATE: It appears that Natus now uses that middle column of 'Duration' for a given clip note name,
    # (they now have the length of duration in minutes), so I'll have to include it when processing the text
    # (i.e., don't remove the double '\t's from the middle

    # manually make the column name row this each time:
    # lines[5] = 'Time\tTitle'

    # remove all extraneous tabs and turn them into a single tab
    # lines = ['\t'.join(line.split('\t\t')) for line in lines]
    # also get rid of weird 'd1 ' (with a space) that was put in front of the times for some of the text files
    # lines = [''.join(line.split('d1 ')) for line in lines]
    # WHELP now apparently some files can have 'd2' in there instead of 'd1'!
    # use a regex on d[number] to remove them.
    lines = [''.join(re.split(r'd\d+ ', line)) for line in lines]

    # only pull non-blank rows.
    # if no description was entered for a timestamp, it will read that row as only having one column and throw an error
    lines = [x for x in lines if '\t' in x]

    # turn into array
    # Now have to specify encoding=None (to use system default) or else it reads as byte-string and puts a b in front.
    table = np.genfromtxt(lines[4:], delimiter='\t', dtype=None, encoding=None)

    # strip whitespace from timestamps
    # as far as I know, the 5th row should always start the columns of time/title data in the text files
    # UPDATE: since I'm now just reading all actual 3 COLUMNS from line 5 onward above,
    # can now just start at the start of 'table'
    for col in range(0, len(table)):
        table[col][0] = table[col][0].strip()

    # get start time
    start = datetime.strptime(table[1][0], '%H:%M:%S')

    # turn timestamps into seconds from start of eeg clip
    for col in range(1, len(table)):
        tmstmp = datetime.strptime(table[col][0], '%H:%M:%S')
        table[col][0] = (tmstmp - start).total_seconds()

    # UPDATE: if table does indeed have 3 columns, remove the middle one ('duration')
    if table.shape[1] == 3:
        # 'obj=1' works to delete the entire column.
        table = np.delete(table, obj=1, axis=1)

    return table


# ==============================================================================
# ## Function to get rows with 'left' and 'right' events from event table
# ==============================================================================

def get_event_rows(event_tab, sfreq):
    labels = []
    # start at 5th row, where all events should always start,
    # to avoid including names with 'right' in them (e.g., 'Wright')
    for row in event_tab:
        # at first, I got around having the useless events by also searching for the word 'hand'
        # but then realized I could just check if 'montage' isn't in the line instead
        # UPDATE: 'hand' needs to be checked for too, since I just found some text files that talk about
        # 'right cheek twitching', and they were being included as events.
        # and 'montage' not in str(row[1]).lower()    and 'montage' not in str(row[1]).lower()
        #  and 'montage' not in str(row[1]).lower()    and 'montage' not in str(row[1]).lower()
        # UPDATE: This shouldn't be checking the second columns of the table, since now the second column
        # could have info in it and can't be stripped out from transform_timestamps.
        # Now the column with the labels is column 3, but really it's just always the last column.
        # Make this instead check row[len(row)-1]
        if (
                ('right' in str(row[len(row) - 1]).lower() and 'hand' in str(row[len(row) - 1]).lower()) or
                ('open' in str(row[len(row) - 1]).lower() and 'right' in str(row[len(row) - 1]).lower()) or

                ('motor' in str(row[len(row) - 1]).lower() and 'right' in str(row[len(row) - 1]).lower()) or
                # I think the events marked w/ an 'r' or 'l' need to be checked w/ a space after the letter,
                # otherwise any word w/ an 'r' or 'l' and has 'open' in it will be counted.
                ('r ' in str(row[len(row) - 1]).lower() and 'hand' in str(row[len(row) - 1]).lower()) or
                ('oc r' in str(row[len(row) - 1]).lower() or 'o/c r' in str(row[len(row) - 1]).lower() or 'oc right' in str(row[len(row) - 1]).lower() or
                 'o/c right' in str(row[len(row) - 1]).lower() or 'oc_right' in str(row[len(row) - 1]).lower() or
                 'oc rh' in str(row[len(row) - 1]).lower() or 'o/c rh' in str(row[len(row) - 1]).lower()) or
                # adding plain 'right' and 'left'. potentially might break older things if 'right' is found in
                # some other word that isn't actually the command?
                ('right' in str(row[len(row) - 1]).lower() and 'montage' not in str(row[len(row) - 1]).lower())
             ):
            labels.append([int(float(row[0])), row[len(row) - 1]])
        if (
                ('left' in str(row[len(row) - 1]).lower() and 'hand' in str(row[len(row) - 1]).lower()) or
                ('open' in str(row[len(row) - 1]).lower() and 'left' in str(row[len(row) - 1]).lower()) or

                ('motor' in str(row[len(row) - 1]).lower() and 'left' in str(row[len(row) - 1]).lower()) or

                ('l ' in str(row[len(row) - 1]).lower() and 'hand' in str(row[len(row) - 1]).lower()) or
                ('oc l' in str(row[len(row) - 1]).lower() or 'o/c l' in str(row[len(row) - 1]).lower() or 'oc left' in str(row[len(row) - 1]).lower() or
                 'o/c left' in str(row[len(row) - 1]).lower() or 'oc_left' in str(row[len(row) - 1]).lower() or
                 'oc lh' in str(row[len(row) - 1]).lower() or 'o/c lh' in str(row[len(row) - 1]).lower()) or
                ('left' in str(row[len(row) - 1]).lower() and 'montage' not in str(row[len(row) - 1]).lower())
             ):
            labels.append([int(float(row[0])), row[len(row) - 1]])

        # still checking for 'alice story' events here, but might not be necessary since
        # all events that don't occur within the specific range of time that the start/stop o/c hand events
        # should occur are set to removed / set to 0 in later functions
        if 'alice' in str(row[len(row) - 1]).lower():
            labels.append([int(float(row[0])), row[len(row) - 1]])

        # need to look for the tennis events too so that I can remove them??
        if 'tennis' in str(row[len(row) - 1]).lower():
            labels.append([int(float(row[0])), row[len(row) - 1]])

    # turn seconds in labels into sampling number
    for row in labels:
        row[0] = row[0] * sfreq

    return labels


# ==============================================================================
# ## Function to remove bad trigs
# ==============================================================================

# So this change does allow for the partially cropped out trigger block at the end of some of the files,
# but in doing so, it also allows for any small trigger blocks of the same size that could be strewn
# throughout the file.
# maybe can add something that will only keep them if they're at the end of the file?
# I think that would entail a whole restructuring of how this function works, though..

def remove_bad_trigs(chan, sec, smallsec, sfreq, min_block_size=16):
    # This removes triggers that aren't in a block of 16 (8 each stop and start).
    # If we don't want that, we'll have to edit this.
    # I think this can be done by just changing the number that 'count' is checked against.
    triggers = np.nonzero(chan[0])

    indices = []
    bad_trigs = []
    # set to stupidly high number to start, so that the conditional check won't "fail" when it checks
    # like it did when this was set to 'None'
    distbehnd = 9999999999999
    for i in range(0, len(triggers[0])):
        if i+1 < len(triggers[0]):
            # calculate distance from current value & next value
            dist = triggers[0][i+1] - triggers[0][i]
            # if distance between values is an acceptable amount
            if dist < sec * sfreq:
                # just add this value to a running list of values
                # which make up the current column
                indices.append(triggers[0][i])

            # if distance between values is greater than an acceptable amount
            if dist > sec * sfreq:
                # add this current value to the running list
                indices.append(triggers[0][i])
                # and see how long the list is
                count = len(indices)

                # if the count of this list is smaller than the amount that,
                # at minimum, should make up an acceptable column
                if count < min_block_size:
                    # print "count is < 16, so removing these: "
                    # print indices
                    # then all these values are bad and are removed by setting to 0
                    for j in indices:
                        bad_trigs.append(j)
                        chan[0][j] = 0
                    # and the running list of values is reset
                    indices = []
                # if the count of this list is larger than (or equal to) the minimum amount,
                # then the values are good. they are left as-is, and the
                # list of values is reset
                if count >= min_block_size:
                    indices = []

            # check the behind distances so that, for any triggers that are too close together,
            # the last one in that group can still be checked
            if i != 0:
                distbehnd = triggers[0][i] - triggers[0][i-1]

            # if distance between values is an acceptable amount
            # and also greater than a stupidly small amount (there are some triggers that are way too close together)
            if dist < sec * sfreq and dist < smallsec * sfreq or distbehnd < sec * sfreq and distbehnd < smallsec * sfreq:
                # checking for the last one in a column w/ too small dist between
                bad_trigs.append(triggers[0][i])
                chan[0][triggers[0][i]] = 0

        # case for last value
        if i == len(triggers[0])-1:
            # since this is the final value, I have to check the one behind it instead
            dist = triggers[0][i] - triggers[0][i-1]
            # if the distance is larger than acceptable, then set it to 0
            if dist > sec * sfreq:
                bad_trigs.append(triggers[0][i])
                chan[0][triggers[0][i]] = 0

            # also need to see if the current column I'm in at the end is < 16 triggers.
            # it's possible that there is a partial column of triggers at the very end

            # if it's not larger than acceptable, check column length
            if dist < sec * sfreq:
                # print "distance is acceptable for the last value. have to check column length"

                indices.append(triggers[0][i])
                count = len(indices)
                if count < min_block_size:
                    for j in indices:
                        bad_trigs.append(j)
                        chan[0][j] = 0

    return chan, bad_trigs


# ==============================================================================
# ## Function to find ranges / thresholds to match the type of event with the trigger
# ==============================================================================

def find_thresh(labels, dtime, sfreq, dtime_add=6):
    thresh = []
    # change end to be 10 minutes after the last event,
    # this way I can remove any events that appear (for some reason) after the last
    # listed event in the text file
    end = labels[len(labels)-1][0]+(600*sfreq)  # 600s == 10 minutes. might want to change this

    # dtime is allowing for some time in-between when they write the event down and the trigger occurring.
    # sometimes the event is written down after the trigger has started

    for i in range(0, len(labels)):
        # get curr val+label
        startval, label = labels[i]

        # allow startval to have some wiggle room beforehand.
        # also allow for a larger start range for the first entry
        if i == 0:
            startval = startval - (dtime+(dtime_add*60)) * sfreq  # allow for dtime_add seconds earlier
            # need to check if starval is negative after this. if so, then set it to 0
            if startval < 0:
                startval = 0
        else:
            startval = startval - dtime * sfreq

        # if a next row, i+1, exists based on the size of the list, get that value
        if i+1 < len(labels):
            endval = labels[i+1][0]

            # also allow for wiggle room after end time?
            endval = endval + dtime * sfreq

        else:
            endval = end

        # create threshold value
        entry = {'label': label, 'range': (startval, endval)}
        # and append it to the threshold list
        thresh.append(entry)

    return thresh


# ==============================================================================
# ## Function to assign values for L hand start and R hand start events
# ==============================================================================
# L hand remains at 1 (default value)
# R hand = 3
def assign_start(trigs, thresh, chan, start_right=3, start_left=1):
    # This will overwrite any bad trigs that were set to 0 to 3 if they technically fall in the event range,
    # even if they aren't legit triggers.
    # set a condition that, if the value == 0, don't overwrite it
    lastline = []   # this is the previous line from the current line
    maxthresh = thresh[len(thresh)-1]['range'][1]
    for sample in trigs:
        for line in thresh:
            # account for overlaps in thresholds first
            if lastline:
                if sample > line['range'][0] and sample < line['range'][1] and \
                        sample > lastline['range'][0] and sample < lastline['range'][1]:  # and chan[0][sample] != 0:

                    if 'right' in str(line['label']).lower() or 'r' in str(line['label']).lower():  # and chan[0][sample] != 0:
                        # print 'assigned right'
                        chan[0][sample] = start_right
                    if 'left' in str(line['label']).lower() or 'l' in str(line['label']).lower():  # and chan[0][sample] != 0:
                        # print 'assigned left'
                        chan[0][sample] = start_left
                    # remove alice and tennis
                    if 'alice' in str(line['label']).lower():
                        chan[0][sample] = 0
                    # if ('tennis' in str(line['label']).lower() and 'right' not in str(lastline['label']).lower()) and
                    # ('tennis' in str(line['label']).lower() and 'left' not in str(lastline['label']).lower()):
                    if 'tennis' in str(line['label']).lower():
                        # print 'removed tennis'
                        chan[0][sample] = 0

            # lastline = line
            # this whole 'left' part is unnecessary, since it's not overwriting anything.
            # leave it just in case I want to make the 'left' values something other than 1.
            # Apparently this part is necessary now that I am checking for overlaps in thresholds.
                elif sample > line['range'][0] and sample < line['range'][1] and chan[0][sample] != 0:

                    if 'left' in str(line['label']).lower() or 'l' in str(line['label']).lower():
                        # print 'assigned left'
                        chan[0][sample] = start_left  # pass
                    if 'right' in str(line['label']).lower() or 'r' in str(line['label']).lower():
                        chan[0][sample] = start_right
                    # remove alice and tennis
                    if 'alice' in str(line['label']).lower():
                        chan[0][sample] = 0
                    if 'tennis' in str(line['label']).lower():
                        chan[0][sample] = 0

            # if lastline is empty, then do this:
            else:
                if sample > line['range'][0] and sample < line['range'][1] and chan[0][sample] != 0:

                    if 'left' in str(line['label']).lower() or 'l' in str(line['label']).lower():
                        chan[0][sample] = start_left  # pass
                    if 'right' in str(line['label']).lower() or 'r' in str(line['label']).lower():
                        chan[0][sample] = start_right
                    # remove alice and tennis
                    if 'alice' in str(line['label']).lower():
                        chan[0][sample] = 0
                    if ('tennis' in str(line['label']).lower()) and ('right' not in str(lastline['label']).lower()):
                        chan[0][sample] = 0

            # remove any events that, for whatever reason, are not listed in the text file
            if sample > maxthresh:
                chan[0][sample] = 0

            lastline = line

    return chan


# ==============================================================================
# ## Function to assign diff values for L hand stop and R hand stop events
# ==============================================================================
# L hand stop = 2
# R hand stop = 4
def assign_stop(trigs, chan, sfreq, t_range, start_right=3, start_left=1, stop_right=4, stop_left=2):
    # t_range is the time allowed between a start/stop event. This will vary for some patients,
    # possibly due to English v. Spanish spoken events taking different amounts of time to say
    for i in range(0, len(trigs)):
        if i+1 < len(trigs):
            # this is essentially the same as checking for bad triggers, except now
            # it just checks if the value is 1 or 3 and sets the value to 2 or 4 instead
            check_front = trigs[i+1] - trigs[i]
            if check_front in range(0, int(t_range * sfreq)):
                if chan[0][trigs[i+1]] == start_left:
                    chan[0][trigs[i+1]] = stop_left

                if chan[0][trigs[i+1]] == start_right:
                    chan[0][trigs[i+1]] = stop_right

    return chan


# ==============================================================================
# ## Make new threshold for removing bad triggers
# ==============================================================================

def make_new_thresh(chan, sfreq, tbuff, start_right=3, start_left=1, stop_right=4, stop_left=2, sndpass=False):
    dist = []
    new_thresh = []
    idx = np.where(chan[0] != 0)
    trgs = chan[0][idx]
    firstofblock = []
    endofblock = []

    for i in range(1, len(idx[0])):
        if i == 1:
            firstofblock = idx[0][i-1]
            curr_avg = idx[0][i]
            if trgs[i-1] == start_left or trgs[i-1] == start_right:   # chan[0][idx[0][i-1]]
                prior_start_trig = trgs[i-1]
            if trgs[i-1] == stop_left or trgs[i-1] == stop_right:
                prior_stop_trig = trgs[i-1]

            if trgs[i] == start_left or trgs[i] == start_right:
                curr_start_trig = trgs[i]
            if trgs[i] == stop_left or trgs[i] == stop_right:
                curr_stop_trig = trgs[i]

            continue

        curr_dist = (idx[0][i] - idx[0][i-1])/sfreq

        # also keep note of what the current start trigger values are so that,
        # if this changes too, it's an indicator of when it's a new block
        # need this because some new blocks start without any gap in time between them.
        if trgs[i-1] == start_left or trgs[i-1] == start_right:
            prior_start_trig = trgs[i-1]
        if trgs[i-1] == stop_left or trgs[i-1] == stop_right:
            prior_stop_trig = trgs[i-1]

        if trgs[i] == start_left or trgs[i] == start_right:
            curr_start_trig = trgs[i]  # chan[0][idx[0][i]]
        if trgs[i] == stop_left or trgs[i] == stop_right:
            curr_stop_trig = trgs[i]

        # find current average so far
        curr_avg = np.mean(dist)

        if sndpass is False:

            # check if current distance value is abnormally larger than the average
            try:
                if (curr_dist > curr_avg + tbuff) or ((curr_start_trig != prior_start_trig) and (((idx[0][i-1] - firstofblock)/sfreq) > 240)):  # turn this to tbuff when done  # give a 20s buffer on the average to check the current distance to?
                    # then this is a large distance and thus is the end of the block.
                    endofblock = idx[0][i-1]
                    # now add the start and end values to new_thresh
                    new_thresh.append((firstofblock, endofblock))
                    # set start of next block
                    firstofblock = idx[0][i]

                    continue

            # have to add this in because, if the first triggers are repeats of a stop trigger,
            # there will be no curr_start_trigger assigned yet for the check to happen.
            except NameError:
                if (curr_dist > curr_avg + tbuff) or (curr_stop_trig != prior_stop_trig):

                    endofblock = idx[0][i-1]
                    new_thresh.append((firstofblock, endofblock))
                    firstofblock = idx[0][i]

                    continue

            if i == len(idx[0])-1:
                endofblock = idx[0][i]
                new_thresh.append((firstofblock, endofblock))

            dist.append(curr_dist)

        # do the same thing, but don't include the check for if the hand-type changes
        if sndpass is True:

            try:
                if curr_dist > curr_avg + tbuff:
                    # then this is a large distance and thus is the end of the block.
                    endofblock = idx[0][i-1]
                    # now add the start and end values to new_thresh
                    new_thresh.append((firstofblock, endofblock))

                    firstofblock = idx[0][i]

                    continue
            # have to add this in because, if the first triggers are repeats of a stop trigger,
            # there will be no curr_start_trigger assigned yet for the check to happen.
            except NameError:
                if curr_dist > curr_avg + tbuff:

                    endofblock = idx[0][i-1]
                    new_thresh.append((firstofblock, endofblock))
                    firstofblock = idx[0][i]

                    continue

            if i == len(idx[0])-1:
                endofblock = idx[0][i]
                new_thresh.append((firstofblock, endofblock))

            dist.append(curr_dist)

    return new_thresh


# ==============================================================================
# # Clean missassigned triggers
# ==============================================================================

def clean_missassigned_trigs(chan, trigs, sfreq, start_right=3, start_left=1, stop_right=4, stop_left=2, btwn_block_time=40):  # new_thresh):

    previous_triggers = []
    for i in range(1, len(trigs)):

        curr_trig = trigs[i]  # chan[0][trigs[i]]
        prev_trig = trigs[i-1]  # chan[0][trigs[i-1]]

        if i == 1:
            previous_triggers.append([chan[0][trigs[i-1]], trigs[i-1]])
            previous_triggers.append([chan[0][trigs[i]], trigs[i]])
            continue

        if (len(previous_triggers) == 16) or ((curr_trig - prev_trig) / sfreq > btwn_block_time):
            # then this is now a new block..
            # this is when the checking for most common trigger type in the block
            just_trigs = [x[0] for x in previous_triggers]
            trig_counts = Counter(just_trigs)
            # this key argument allows for getting the key out of this dictionary with the corresponding highest value
            most_trig = max(trig_counts, key=trig_counts.get)

            if most_trig == start_left or most_trig == stop_left:
                # get incorrectly labeled ones
                # python3 needs filter wrapped by something, like list, to extract the elements from it now
                bad_label_trigs = list(filter(lambda x: x[0] == start_right or x[0] == stop_right, previous_triggers))
                if len(bad_label_trigs) != 0:
                    for trg in bad_label_trigs:
                        if trg[0] == start_right:
                            chan[0][trg[1]] = start_left
                        if trg[0] == stop_right:
                            chan[0][trg[1]] = stop_left

            if most_trig == start_right or most_trig == stop_right:
                bad_label_trigs = list(filter(lambda x: x[0] == start_left or x[0] == stop_left, previous_triggers))
                if len(bad_label_trigs) != 0:
                    for trg in bad_label_trigs:
                        if trg[0] == start_left:
                            chan[0][trg[1]] = start_right
                        if trg[0] == stop_left:
                            chan[0][trg[1]] = stop_right

            previous_triggers = []

        else:
            # always append both the trigger and its index together
            previous_triggers.append( [chan[0][trigs[i]], trigs[i]] )

    return chan


# ==============================================================================
# ###### make a new function to fix misassigned trigs,
# ###### but have it use new_thresh because it should be easier
# I need both clean_missassigned_trigs functions because the first one
# works without the assumption of clearly defined trigger blocks.
# After all is said and done, this one can be used to clean up any remaining ones.
# ==============================================================================

def clean_missassigned_trigs2(chan, new_thresh, start_right=3, start_left=1, stop_right=4, stop_left=2):
    for i in range(0, len(new_thresh)):
        trgs = chan[0][new_thresh[i][0]:new_thresh[i][1]+1]
        trgs_c = Counter(trgs)
        # remove the 0s from the counter. "0.0" is the key in the counter dictionary
        del trgs_c[0.0]
        # find trigger that appears the most in this block
        mx_trg = max(trgs_c, key=trgs_c.get)
        # reassign like this
        if mx_trg == start_left or mx_trg == stop_left:
            trgs = [start_left if x == start_right else x for x in trgs]
            trgs = [stop_left if x == stop_right else x for x in trgs]
        if mx_trg == start_right or mx_trg == stop_right:
            trgs = [start_right if x == start_left else x for x in trgs]
            trgs = [stop_right if x == stop_left else x for x in trgs]

        # then reassign this block
        chan[0][new_thresh[i][0]:new_thresh[i][1]+1] = trgs

    return chan


# ==============================================================================
# ## Function to remove duplicate triggers after bads have been removed and start/stop assigned
# ==============================================================================

def clean_trigger_blocks(chan, sfreq, new_thresh, start_right=3, start_left=1, stop_right=4, stop_left=2):

    # if new_thresh was able to be made
    for line in new_thresh:
        # have to add and subtract 1 to the ends of the thresholds,
        # since the end ranges ARE the actual indices of the first and last points,
        # so they won't be counted in the unique function if the end range == the last point.
        unique, counts = np.unique(chan[0][int(line[0]-1):int(line[1]+1)], return_counts=True)

        if len(unique) == 1 and unique[0] == 0:
            print('No triggers found.')

        else:
            countdic = dict(zip(unique, counts))

        # new method:
        for key in countdic.keys():
            if key == 0:
                continue
            if countdic[key] > 8:
                # find what the triggers in this block, excluding 0
                trigs_in_block = list(filter(lambda x: x != 0, countdic.keys() ))

                # need to index entirety of chan.
                # Plan is to count backwards from the number of triggers in this block
                # (assuming the last triggers are assigned to the correct hand),
                # then remove any left after counting 16 of them.
                all_idx_trig = [i for i, x in enumerate(chan[0]) if x in trigs_in_block] # == 3 or x == 4]

                # and then filter that between the start and end indices of this block
                idx_of_all_idx = np.where(np.logical_and(all_idx_trig >= line[0], all_idx_trig <= line[1]))
                # then use these indices to get the indices from 'aa' which are the indices from chan
                # that relate to this block.
                block_idx_trig = [all_idx_trig[i] for i in idx_of_all_idx[0]]

                # now go through each index in reverse and count how many of each trigger there are.
                # once the current trigger is the same as the previous one,
                # remove all remaining triggers and then set the current trigger as a start event.
                rev = reversed(block_idx_trig)
                rev_as_lst = [i for i in rev]

                # here's where the actual removal occurs
                trig_counts = [(value, key) for key, value in countdic.items()]
                trig_counts_no_zero = list(filter(lambda x: x[1] != 0, trig_counts))
                event_w_most_trigs = max(trig_counts_no_zero)[1]

                for i in range(1, len(rev_as_lst)):
                    prev_trig = chan[0][rev_as_lst[i-1]]
                    curr_trig = chan[0][rev_as_lst[i]]

                    if curr_trig == prev_trig:
                        # if current trigger is a stop trigger,
                        if curr_trig == stop_left or curr_trig == stop_right:
                            # set all triggers before it to 0
                            chan[0][rev_as_lst[i+1:]] = 0
                        # if current trigger is a start trigger,
                        if curr_trig == start_left or curr_trig == start_right:
                            # set all triggers before, and including this one, to 0.
                            chan[0][rev_as_lst[i:]] = 0

                        # then change the current one to the corresponding start trigger
                        # for this event block (L or R hand)
                        # if count of trig is > 1 or 2 or something (to know it's not just an extra one..)
                        # better yet, just find the trigger/key in dic that has the highest count.
                        # if that's a R hand one, then set this to a start right. If it's left, set it to start left.
                        # only set it, though, if it's not already a start trigger
                        if curr_trig == stop_left or curr_trig == stop_right:
                            if event_w_most_trigs == start_right or event_w_most_trigs == stop_right:
                                chan[0][rev_as_lst[i]] = start_right

                            if event_w_most_trigs == start_left or event_w_most_trigs == stop_left:
                                chan[0][rev_as_lst[i]] = start_left

                        break

    return chan


def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.mean(d)
    s = d/mdev if mdev else 0.
    return data[s < m]


# ==============================================================================
# ## shift events function
# ==============================================================================

def shift_events(events, sfreq, is_control, start_right=4, start_left=2, stop_right=5, stop_left=3):
    def shift_notmove(move, notmove, sfreq):
        arr = []
        for i in range(1, len(move)):
            arr.append((move[i] - notmove[i - 1]) - (15 * sfreq))
        out = round(np.median(reject_outliers(np.array(arr))))

        return out

    def shift_move(move, notmove, sfreq):
        arr = []
        for i in range(0, len(notmove)):
            arr.append((notmove[i] - move[i]) - (10 * sfreq))
        out = round(np.median(arr))

        return out

    # Can do this by converting the events array to a pandas dataframe
    events_df = pd.DataFrame(events, columns=['samp', 'na', 'trig'], copy=True)
    # find where all 'keep moving right hand' triggers occur
    kmR = events_df['samp'][events_df['trig']==start_right]
    # find where all 'stop moving right hand' triggers occur
    smR = events_df['samp'][events_df['trig']==stop_right]
    # find where all 'keep moving left hand' triggers occur
    kmL = events_df['samp'][events_df['trig']==start_left]
    # find where all 'stop moving left hand' triggers occur
    smL = events_df['samp'][events_df['trig']==stop_left]
    # the least headache-inducing method to subtract these two "columns" is to convert them to numpy arrays,
    # subtract them, then convert it all back to a pandas dataframe.
    smR_np = np.array(smR)
    kmR_np = np.array(kmR)
    kmL_np = np.array(kmL)
    smL_np = np.array(smL)

    if (len(smL_np) == 0 or len(smR_np) == 0) and not is_control:
        print('file only has left or right hand events--canceling shift')
        return None

    if len(kmR_np) > 0 and len(smR_np) > 0:
        kri_med = shift_move(move=kmR_np, notmove=smR_np, sfreq=sfreq)
        sri_med = shift_notmove(move=kmR_np, notmove=smR_np, sfreq=sfreq)
    else:
        kri_med = np.nan
        sri_med = np.nan

    if len(kmL_np) > 0 and len(smL_np) > 0:
        kli_med = shift_move(move=kmL_np, notmove=smL_np, sfreq=sfreq)
        sli_med = shift_notmove(move=kmL_np, notmove=smL_np, sfreq=sfreq)
    else:
        kli_med = np.nan
        sli_med = np.nan

    # just return all these values for now
    # return kri_avg, sri_avg, kli_avg, sli_avg
    # test editing events by converting to pandas df and then back to np array

    if not np.isnan(kri_med):
        newkmR = kmR + np.int64(kri_med)
        newsmR = smR + np.int64(sri_med)
        events_df.loc[events_df['trig'] == start_right, 'samp'] = newkmR
        events_df.loc[events_df['trig'] == stop_right, 'samp'] = newsmR

    if not np.isnan(kli_med):
        newkmL = kmL + np.int64(kli_med)
        newsmL = smL + np.int64(sli_med)
        events_df.loc[events_df['trig'] == start_left, 'samp'] = newkmL
        events_df.loc[events_df['trig'] == stop_left, 'samp'] = newsmL

    new_events = np.array(events_df)

    return new_events


def compute_norm_command_len(events, sfreq, fname1, kri_list, sri_list, kli_list, sli_list, start_right=4, start_left=2, stop_right=5, stop_left=3):
    events_df = pd.DataFrame(events, columns=['samp', 'na', 'trig'])
    # find where all 'keep moving right hand' triggers occur
    kmR = events_df['samp'][events_df['trig'] == start_right]
    # find where all 'stop moving right hand' triggers occur
    smR = events_df['samp'][events_df['trig'] == stop_right]
    # find where all 'keep moving left hand' triggers occur
    kmL = events_df['samp'][events_df['trig'] == start_left]
    # find where all 'stop moving left hand' triggers occur
    smL = events_df['samp'][events_df['trig'] == stop_left]
    # the least headache-inducing method to subtract these two "columns" is to convert them to numpy arrays,
    # subtract them, then convert it all back to a pandas dataframe.
    smR_np = np.array(smR)
    kmR_np = np.array(kmR)
    kmL_np = np.array(kmL)
    smL_np = np.array(smL)

    # Right hand
    kri = []
    for i in range(0, len(smR_np)):
        kri.append( (smR_np[i] - kmR_np[i]) - (10 * sfreq) )
        # this works for kri.

    # it = iter(kmR_np)
    sri = []
    for i in range(1, len(kmR_np)):
        sri.append((kmR_np[i] - smR_np[i-1]) - (15 * sfreq))
        # this includes the huge gap that might be present between different runs of R to L...
    # get rid of outlier

    # Left hand
    kli = []
    for i in range(0, len(smL_np)):
        kli.append((smL_np[i] - kmL_np[i]) - (10 * sfreq))

    sli = []
    for i in range(1, len(kmL_np)):
        sli.append((kmL_np[i] - smL_np[i-1]) - (15 * sfreq))

    if np.isnan(np.mean(kri)) is not True:
        [kri_list.append(x) for x in kri - np.int64(round(np.mean(kri)))]
        [sri_list.append(x) for x in sri - np.int64(round(np.mean(reject_outliers(np.array(sri)))))]
    if np.isnan(np.mean(kli)) is not True:
        [kli_list.append(x) for x in kli - np.int64(round(np.mean(kli)))]
        [sli_list.append(x) for x in sli - np.int64(round(np.mean(reject_outliers(np.array(sli)))))]


# ==============================================================================
# ## Function to remove amplitudes greater than a specified value. Defaults to 120uV
# ==============================================================================
# if 0 < n_comp < 1, then it is the cumulative percentage of explained variance used to choose components.
def remove_artifacts(raw, picks, n_comp=0.95, thresh=240e-6):
    ica = mne.preprocessing.ICA(n_components=n_comp, method='fastica')
    ica.fit(raw, picks=picks, reject=dict(eeg=thresh))

    raw_removed = ica.apply(raw)

    return raw_removed


# =============================================================================
# # Function to find neighboring channels
# ==============================================================================
# Save nearest-check results into new dictionary
def find_nearest_chans(mont, use_ch):
    # get indices of use_ch that are in the channel names of the montage
    indices = mne.pick_channels(mont.ch_names, include=use_ch)

    nearest = dict()
    for i in indices:
        dists = dict()

        for j in indices:
            chans = mont.ch_names[j]

            x1, x2 = mont.pos[i][0], mont.pos[j][0]
            y1, y2 = mont.pos[i][1], mont.pos[j][1]
            z1, z2 = mont.pos[i][2], mont.pos[j][2]
            # double asterisk is raise to power
            dist = np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )
            dists[chans] = dist
            if j == indices[-1]:
                # this gives the 6 closest chans.
                # start at 1 to avoid adding the comparison of the same channel to itself
                small_ds = (sorted(dists.iteritems(), key=operator.itemgetter(1), reverse=False)[1:7])
                # can assign this output, and then iterate over it and do windowing-stats for just these channels.
                # or just make a dict with the use_ch[i]th name as key and the n-th closest chans as the value,
                # and then iterate the values for each key
                nearest[mont.ch_names[i]] = [x[0] for x in small_ds]

    return nearest


# ==============================================================================
# # Function to find bad (and flat) channels
# ==============================================================================
# Use the 'nearest' dictionary in the window-checking to compare each chan to only its neighbors
# initialize start index to 0
def find_bad_chans(raw, nearest, use_ch, per_bad=1.7, per_flat=0.4, wins=5, min_num_bad=2):
    # generate window lengths based on total length of data
    # total len
    d_len = len(raw._data[0])
    # wins == number of windows
    # len of window segments
    seg_len = d_len / wins

    # all ch names
    all_names = raw.info['ch_names']

    # channel indices
    picks = mne.pick_channels(raw.info['ch_names'], use_ch)

    start = 0
    potential_bads = []
    potential_flat = []
    # For each window segment,
    for i in range(0, wins):
        # set the start:stop of the window
        start = start
        stop = start + seg_len
        # and then for each channel,
        # (this is using the indices that correspond to the channels of interest in the raw data)
        # right now it's printing the max/sd for each channel for each window.
        # I need it to then compare these values to each channel's neighbors still for each window
        for j in picks:
            # find the values for the current window
            # can use 'all_names[j]' as the key for the nearest dictionary
            close_chans_bads = nearest[all_names[j]][0:4]  # take the 4 nearest neighbors
            close_chans_flat = nearest[all_names[j]]  # take all 6 nearest neighbors
            # get index of current chan
            # 70% more than current channel's sd:
            curr_chan_sd = raw._data[j, start:stop].std() # * 1.7
            # make a counter of how many times it was greater
            count_bads = 0
            count_flat = 0

            # should make new_pick out here and then iterate over those
            # I guess this way didn't matter here since the indices would have been the same
            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_bads)

            for k in range(0, len(new_pick)):
                val = raw._data[new_pick[k], start:stop].std() * per_bad
                # val corresponds to the neighbor-channel.
                # if current channel is larger than its 70%-increased-neighbors 2 or more times, then it's probably bad
                if val < curr_chan_sd:
                    count_bads += 1
                if k == len(close_chans_bads)-1:
                    # print count
                    # checking if was greater than half of the number of 4th nearest channels (i.e., 2)
                    # don't do greater or equal to
                    if count_bads >= len(close_chans_bads) / 2:
                        potential_bads.append(all_names[j])

            # Do same thing to find flat channels, but use all 6 nearest neighbors

            # should make new_pick out here and then iterate over those
            new_pick2 = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for x in range(0, len(new_pick2)):
                val = raw._data[new_pick2[x], start:stop].std() * per_flat

                if val > curr_chan_sd:
                    count_flat += 1
                if x == len(close_chans_flat)-1:
                    # check if was less than half the number of 6th nearest channels (i.e., 3)
                    # don't do greater or equal to
                    if count_flat >= len(close_chans_flat) / 2:
                        potential_flat.append(all_names[j])

        start = stop

    # find count of potential_bads
    bad_chan_counts = Counter(potential_bads)
    # find count of potential flats
    flat_chan_counts = Counter(potential_flat)

    # find instances where channel was bad in > 2 windows out of 5 using 'dictionary comprehension' syntax
    actually_bad_chans = {k:v for (k, v) in bad_chan_counts.items() if v >= min_num_bad}
    actually_flat_chans = {k:v for (k, v) in flat_chan_counts.items() if v >= min_num_bad}
    # extract channel names
    bad_chan_names = actually_bad_chans.keys()
    flat_chan_names = actually_flat_chans.keys()

    # return actually_bad_chans.keys()
    # turn to set and then back to list to remove any duplicates just in case a channel could have been bad and a flat?
    # return list(set(bad_chan_names + flat_chan_names))
    # return both bad chans and flat chans seperate
    return bad_chan_names, flat_chan_names


# Find bad chans but use 3 minute window
def other_find_bad_chans(raw, nearest, use_ch, per_bad=1.7, per_flat=0.4, minute_blocks=3, min_num_bad=1):

    # all ch names
    all_names = raw.info['ch_names']

    # channel indices
    picks = mne.pick_channels(raw.info['ch_names'], use_ch)

    d_len = len(raw._data[0])

    samples_in_blocks = minute_blocks * 60 * raw.info['sfreq']

    number_of_blocks = int(np.ceil(d_len / samples_in_blocks))

    # bad if > than 1 or more 'parts'
    # min_num_bad = 1

    start = 0
    potential_bads = []
    potential_flat = []
    # For each window segment,
    for i in range(number_of_blocks):
        # set the start:stop of the window
        start = start
        stop = start + samples_in_blocks
        if stop > d_len:
            stop = d_len

        # print start, stop
        # and then for each channel,
        # (this is using the indices that correspond to the channels of interest in the raw data)
        # right now it's printing the max/sd for each channel for each window.
        # I need it to then compare these values to each channel's neighbors still for each window
        for j in picks:
            print("curr chan: " + all_names[j])
            # find the values for the current window
            print(start, stop)
            # can use 'all_names[j]' as the key for the nearest dictionary
            close_chans_bads = nearest[all_names[j]][0:4]  # take the 4 nearest neighbors
            close_chans_flat = nearest[all_names[j]]  # take all 6 nearest neighbors
            # get index of current chan
            # 70% more than current channel's sd:
            curr_chan_sd = raw._data[j, start:stop].std() # * 1.7
            print ('curr chan sd: ' + str(curr_chan_sd))
            # make a counter of how many times it was greater
            count_bads = 0
            count_flat = 0

            # should make new_pick out here and then iterate over those
            # I guess this way didn't matter here since the indices would have been the same
            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_bads)

            for k in range(0, len(new_pick)):  # len(close_chans_bads)):
                val = raw._data[new_pick[k], start:stop].std() * per_bad
                # val corresponds to the neighbor-channel.
                # if current channel is larger than its 70%-increased-neighbors 2 or more times, then it's probably bad
                # print curr_chan_sd, val
                if val < curr_chan_sd:
                    count_bads += 1
                    print(close_chans_bads[k])
                    print('neighb chan sd+70% is less: ' + str(val))
                if k == len(close_chans_bads)-1:
                    print (count_bads)

                    # checking if was greater than half of the number of 4th nearest channels (i.e., 2)
                    # don't do greater or equal to
                    if count_bads >= len(close_chans_bads) / 2:  # maybe these do need to be greater or equal.
                        potential_bads.append(all_names[j])

            # should make new_pick out here and then iterate over those
            new_pick2 = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for x in range(0, len(new_pick2)):
                val = raw._data[new_pick2[x], start:stop].std() * per_flat
                # print 'neighb chan 20% of for flat: ' + str(val)
                # print curr_chan_sd, val

                if val > curr_chan_sd:
                    count_flat += 1
                    print(close_chans_flat[x])
                    print('40% of neighb chan is greater: ' + str(val))
                if x == len(close_chans_flat)-1:
                    # check if was less than half the number of 6th nearest channels (i.e., 3)
                    # don't do greater or equal to
                    if count_flat >= len(close_chans_flat) / 2:
                        potential_flat.append(all_names[j])

            # Then have to check the max & sd for all channels nearest the current channel being interated
        # then, once that's done for each channel, update start index
        # and do this again for the next window
        start = stop

    # find count of potential_bads
    bad_chan_counts = Counter(potential_bads)
    # find count of potential flats
    flat_chan_counts = Counter(potential_flat)

    # find instances where channel was bad in > 2 windows out of 5 using 'dictionary comprehension' syntax
    actually_bad_chans = {k:v for (k, v) in bad_chan_counts.items() if v >= min_num_bad}
    actually_flat_chans = {k:v for (k, v) in flat_chan_counts.items() if v >= min_num_bad}
    # extract channel names
    bad_chan_names = actually_bad_chans.keys()
    flat_chan_names = actually_flat_chans.keys()

    return bad_chan_names, flat_chan_names


# nearest & bad chan funcs rewritten to use picks instead of list of channel names
def find_nearest_chans_2(mont, use_ch):
    # get indices of use_ch that are in the channel names of the montage
    indices = mne.pick_channels(mont.ch_names, include=use_ch)

    nearest = dict()
    for i in indices:
        dists = dict()

        for j in indices:
            # really just need to put the comparison channel in the key
            # chans = use_ch[j] #(use_ch[i], use_ch[j])
            # chans should be pulled from mont.ch_names with these indices
            chans = mont.ch_names[j]

            x1, x2 = mont.pos[i][0], mont.pos[j][0]
            y1, y2 = mont.pos[i][1], mont.pos[j][1]
            z1, z2 = mont.pos[i][2], mont.pos[j][2]
            # double asterisk is raise to power
            dist = np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )
            dists[chans] = dist
            if j == indices[-1]:
                # this gives the 6 closest chans.
                # start at 1 to avoid adding the comparison of the same channel to itself
                small_ds = (sorted(dists.iteritems(), key=operator.itemgetter(1), reverse=False)[1:7])
                # can assign this output, and then iterate over it and do windowing-stats for just these channels.
                # ...or just make a dict with the use_ch[i]th name as key and the n-th closest chans as the value,
                # and then iterate the values for each key
                nearest[mont.ch_names[i]] = [x[0] for x in small_ds]

    return nearest


# Find bad chans but use 3 minute window
def other_find_bad_chans_2(raw, nearest, use_ch, per_bad=1.7, per_flat=0.4, minute_blocks=3, min_num_bad=1):

    # all ch names
    all_names = raw.info['ch_names']

    # channel indices
    picks = mne.pick_channels(raw.info['ch_names'], use_ch)

    d_len = len(raw._data[0])

    samples_in_blocks = minute_blocks * 60 * raw.info['sfreq']

    number_of_blocks = int(np.ceil(d_len / samples_in_blocks))

    # bad if > than 1 or more 'parts'
    # min_num_bad = 1

    start = 0
    potential_bads = []
    potential_flat = []
    # For each window segment,
    for i in range(number_of_blocks):
        # set the start:stop of the window
        start = start
        stop = start + samples_in_blocks
        if stop > d_len:
            stop = d_len

        # print start, stop
        # and then for each channel,
        for j in picks:
            print ("curr chan: " + all_names[j])
            # find the values for the current window
            print (start, stop)
            # can use 'all_names[j]' as the key for the nearest dictionary
            close_chans_bads = nearest[all_names[j]][0:4]  # take the 4 nearest neighbors
            close_chans_flat = nearest[all_names[j]]  # take all 6 nearest neighbors
            # get index of current chan
            # 70% more than current channel's sd:
            curr_chan_sd = raw._data[j, start:stop].std()  # * 1.7
            print ('curr chan sd: ' + str(curr_chan_sd))
            # make a counter of how many times it was greater
            count_bads = 0
            count_flat = 0

            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_bads)

            for k in range(0, len(new_pick)):
                val = raw._data[new_pick[k], start:stop].std() * per_bad
                # print 'neighb chan sd+70% for bad: ' + str(val)
                # val corresponds to the neighbor-channel.
                # if current channel is larger than its 70%-increased-neighbors 2 or more times, then it's probably bad
                # print curr_chan_sd, val
                if val < curr_chan_sd:
                    count_bads += 1
                    print(close_chans_bads[k])
                    print('(neighb chan sd+70% is less: ' + str(val))
                if k == len(close_chans_bads)-1:
                    print(count_bads)

                    # checking if was greater than half of the number of 4th nearest channels (i.e., 2)
                    # don't do greater or equal to
                    if count_bads >= len(close_chans_bads) / 2:  # maybe these do need to be greater or equal.
                        potential_bads.append(all_names[j])

            new_pick2 = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for x in range(0, len(new_pick2)):
                # check if < 20% of at least 3 of 6 neighbors
                val = raw._data[new_pick2[x], start:stop].std() * per_flat

                if val > curr_chan_sd:
                    count_flat += 1
                    print(close_chans_flat[x])
                    print('40% of neighb chan is greater: ' + str(val))
                if x == len(close_chans_flat)-1:
                    # check if was less than half the number of 6th nearest channels (i.e., 3)
                    # don't do greater or equal to
                    if count_flat > len(close_chans_flat) / 2:
                        potential_flat.append(all_names[j])

        start = stop

    # find count of potential_bads
    bad_chan_counts = Counter(potential_bads)
    # find count of potential flats
    flat_chan_counts = Counter(potential_flat)

    # find instances where channel was bad in > 2 windows out of 5 using 'dictionary comprehension' syntax
    actually_bad_chans = {k:v for (k,v) in bad_chan_counts.items() if v >= min_num_bad}
    actually_flat_chans = {k:v for (k,v) in flat_chan_counts.items() if v >= min_num_bad}
    # extract channel names
    bad_chan_names = actually_bad_chans.keys()
    flat_chan_names = actually_flat_chans.keys()

    return bad_chan_names, flat_chan_names


# ==============================================================================
# # #### Find flat chans first, then reject them from the neighbor comparison
# ==============================================================================
def find_flat_chans(raw, nearest, use_ch, per_flat=0.4, minute_blocks=3, min_num_bad=1):
    # all ch names
    all_names = raw.info['ch_names']
    # channel indices
    picks = mne.pick_channels(raw.info['ch_names'], use_ch)
    # length of data
    d_len = len(raw._data[0])
    # number of samples in this time window
    samples_in_blocks = minute_blocks * 60 * raw.info['sfreq']
    # number of time blocks there are
    number_of_blocks = int(np.ceil(d_len / samples_in_blocks))

    start = 0
    potential_flat = []
    # For each window segment,
    for i in range(number_of_blocks):
        # set the start:stop of the window
        start = int(start)
        stop = int(start + samples_in_blocks)
        if stop > d_len:
            stop = int(d_len)

        for j in picks:
            close_chans_flat = nearest[all_names[j]]  # take all 6 nearest neighbors
            # get index of current chan
            # 70% more than current channel's sd:
            curr_chan_sd = raw._data[j, start:stop].std() #* 1.7
            # make a counter of how many times it was greater
            count_flat = 0
            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for k in range(0, len(new_pick)):

                val = raw._data[new_pick[k], start:stop].std() * per_flat
                # val corresponds to the neighbor-channel.
                # if current channel is larger than its 70%-increased-neighbors 2 or more times, then it's probably bad
                # print curr_chan_sd, val
                if val > curr_chan_sd:
                    count_flat += 1
                if k == len(close_chans_flat)-1:
                    # checking if was greater than half of the number of 4th nearest channels (i.e., 2)
                    # don't do greater or equal to
                    if count_flat > len(close_chans_flat) / 2:  # maybe these do need to be greater or equal.
                        potential_flat.append(all_names[j])

        start = stop

    # find count of potential flats
    flat_chan_counts = Counter(potential_flat)
    # find instances where channel was bad in > 2 windows out of 5 using 'dictionary comprehension' syntax
    actually_flat_chans = {k: v for (k, v) in flat_chan_counts.items() if v >= min_num_bad}
    # extract channel names
    flat_chan_names = actually_flat_chans.keys()

    return flat_chan_names


def find_bad_chans_final(raw, nearest, use_ch, per_bad=1.7, minute_blocks=3, min_num_bad=1):
    print('checking for bad channels...')
    # all ch names
    all_names = raw.info['ch_names']
    # channel indices
    picks = mne.pick_channels(raw.info['ch_names'], use_ch)
    # length of data
    d_len = len(raw._data[0])
    # number of samples in this time window
    samples_in_blocks = minute_blocks * 60 * raw.info['sfreq']
    # number of time blocks there are
    number_of_blocks = int(np.ceil(d_len / samples_in_blocks))
    # find flat channels
    flat_chans = find_flat_chans(raw, nearest, use_ch, per_flat=0.4,
                                 minute_blocks=minute_blocks, min_num_bad=min_num_bad)

    start = 0
    potential_bad = []
    # For each window segment,
    for i in range(number_of_blocks):
        # set the start:stop of the window
        # make these ints b/c it now gives warning error about using floats to index.
        start = int(start)
        stop = int(start + samples_in_blocks)
        if stop > d_len:
            stop = int(d_len)

        for j in picks:
            print ("curr chan: " + all_names[j])
            # find the values for the current window
            close_chans_bad = nearest[all_names[j]][0:4]  # take 4 neighbors
            # get index of current chan
            # 70% more than current channel's sd:
            curr_chan_sd = raw._data[j, start:stop].std() # * 1.7
            # make a counter of how many times it was greater
            count_bad = 0
            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_bad, exclude=flat_chans)

            for k in range(0, len(new_pick)):

                val = raw._data[new_pick[k], start:stop].std() * per_bad
                # val corresponds to the neighbor-channel.
                # if current channel is larger than its 70%-increased-neighbors 2 or more times, then it's probably bad
                if val < curr_chan_sd:
                    count_bad += 1
                if k == len(close_chans_bad)-1:
                    # checking if was greater than half of the number of 4th nearest channels (i.e., 2)
                    # don't do greater or equal to
                    if count_bad > len(close_chans_bad) / 2:
                        potential_bad.append(all_names[j])

        start = stop

    # find count of potential bads
    bad_chan_counts = Counter(potential_bad)
    # find instances where channel was bad in > 2 windows out of 5 using 'dictionary comprehension' syntax
    actually_bad_chans = {k:v for (k,v) in bad_chan_counts.items() if v >= min_num_bad}
    # extract channel names
    bad_chan_names = actually_bad_chans.keys()

    return bad_chan_names, flat_chans


def get_event_rows_healthy_volun(event_tab, sfreq):
    labels = []
    # start at 5th row, where all events should always start,
    # to avoid including names with 'right' in them (e.g., 'Wright')
    for row in event_tab[5:]:
        # at first, I got around having the useless events by also searching for the word 'hand'
        # but then realized I could just check if 'montage' isn't in the line instead
        if 'right' in str(row[1]).lower() and 'montage' not in str(row[1]).lower():
            labels.append([int(float(row[0])), row[1]])
        if 'left' in str(row[1]).lower() and 'montage' not in str(row[1]).lower():
            labels.append([int(float(row[0])), row[1]])

        # still checking for 'alice story' events here, but might not be necessary since
        # all events that don't occur within the specific range of time that the start/stop o/c hand events
        # should occur are set to removed / set to 0 in later functions
        if 'alice' in str(row[1]).lower():
            labels.append([int(float(row[0])), row[1]])

    # turn seconds in labels into sampling number
    for row in labels:
        row[0] = row[0] * sfreq

    return labels


# ==============================================================================
# ## Function to assign values for L hand start and R hand start events,
# ## modified to get a baseline for 'stop moving' -- trigger for the 'stop moving' is moved
# ## 2.5 seconds further so that the 'stop moving' baseline doesn't occur during the end of
# ## the 'start moving' command
# ==============================================================================
# L hand remains at 1 (default value)
# R hand = 3
def assign_start_mod_for_baseln(trigs, thresh, chan, start_right=3, start_left=1):
    # This will overwrite any bad trigs that were set to 0 to 3 if they technically fall in the event range,
    # even if they aren't legit triggers.
    #  set a condition that, if the value == 0, don't overwrite it
    lastline = []   # this is the previous line from the current line
    maxthresh = thresh[len(thresh)-1]['range'][1]
    for sample in trigs:
        for line in thresh:
            # account for overlaps in thresholds first
            if lastline != []:
                if sample > line['range'][0] and sample < line['range'][1] and sample > lastline['range'][0] and \
                        sample < lastline['range'][1] and chan[0][sample] != 0:
                    if 'right' in str(line['label']).lower():
                        chan[0][sample] = start_right
                    if 'left' in str(line['label']).lower():
                        chan[0][sample] = start_left
            # this whole 'left' part is unnecessary, since it's not overwriting anything.
            # I guess leave it just in case I want to make the 'left' values something other than 1.
            # Apparently this part is necessary now that I am checking for overlaps in thresholds.
                elif sample > line['range'][0] and sample < line['range'][1] and chan[0][sample] != 0:
                    if 'left' in str(line['label']).lower():
                        chan[0][sample] = start_left
                    if 'right' in str(line['label']).lower():
                        chan[0][sample] = start_right

            else:
                if sample > line['range'][0] and sample < line['range'][1] and chan[0][sample] != 0:
                    if 'left' in str(line['label']).lower():
                        chan[0][sample] = start_left
                    if 'right' in str(line['label']).lower():
                        chan[0][sample] = start_right

            # remove any events that, for whatever reason, are not listed in the text file
            if sample > maxthresh:
                chan[0][sample] = 0

            lastline = line

    return chan


# ==============================================================================
# ## Function to assign diff values for L hand stop and R hand stop events.
# ## # ## modified to get a baseline for 'stop moving' -- trigger for the 'stop moving' is moved
# ## 2.5 seconds further so that the 'stop moving' baseline doesn't occur during the end of
# ## the 'start moving' command
# ==============================================================================
# L hand stop = 2
# R hand stop = 4
def assign_stop_mod_for_baseln(trigs, chan, sfreq, t_range, start_right=3, start_left=1, stop_right=4, stop_left=2):
    for i in range(0, len(trigs)):
        if i+1 < len(trigs):
            # this is essentially the same as checking for bad triggers, except now
            # it just checks if the value is 1 or 3 and sets the value to 2 or 4 instead
            check_front = trigs[i+1] - trigs[i]
            if check_front in range(0, int(t_range * sfreq)):
                if chan[0][trigs[i+1]] == start_left:
                    chan[0][trigs[i+1]] = stop_left

                if chan[0][trigs[i+1]] == start_right:
                    chan[0][trigs[i+1]] = stop_right

    return chan


def clean_trigger_block_id_pair(labs, l_ids=[10,20,30,40], r_ids=[50,60,70,80]):
    l_change = []
    r_change = []
    for i in range(1, len(labs)):
        if (labs[i - 1] not in l_ids) and (labs[i] in l_ids):
            l_change.append(i)
        if (labs[i - 1] not in r_ids) and (labs[i] in r_ids):
            r_change.append(i)

    new_block_id = np.array(l_change + r_change)
    new_block_id.sort()

    return new_block_id


def find_event_block_boundaries(events, dist_thresh):
    labs = events[:, 2]
    # index differences
    diffs = np.diff(events[:, 0])
    # gaps where block changes based on distance
    # add 1 to the indices to get the start of the next block and not the end of the previous one
    new_block_dist = np.where(diffs > dist_thresh)[0] + 1

    new_block_id = clean_trigger_block_id_pair(labs=labs, l_ids=[10, 20, 30, 40], r_ids=[50, 60, 70, 80])

    new_block_idxs = np.unique(np.concatenate((new_block_dist, new_block_id), axis=None))
    # add in the start and end
    new_block_idxs2 = np.concatenate((new_block_idxs, (0, len(labs)))).astype(int)
    new_block_idxs2.sort()

    return new_block_idxs2


def clean_trigger_blocks2(events, dist_thresh=10000, min_block_size=12, max_block_size=32):

    new_block_idxs2 = find_event_block_boundaries(events, dist_thresh=dist_thresh)

    # count number of event ids between the new block idx
    # and pick out blocks that are too small and should be removed
    # it might not be necessary, but as of now this part won't work when events are already
    #  in 4 different IDs per block.
    #  It checks the length of the block by looking at all IDs in it--but wait
    #  that's exactly the same as doing it on the non-dc7'd blocks so I guess it's fine after all
    #  so replace _check_events's block definition with this and then see how it works
    #  to insert/remove events after having run clean_events and clean_trigger_blocks2
    #  maybe make the above code to define new_block_idxs2 into its own function as well
    #  that way I can use that in _check_events and then just copy the first for loop below
    #  into _check_events to subset the actual blocks.
    rm_idxs = []
    block_sizes = []
    for i in range(1, len(new_block_idxs2)):
        start = new_block_idxs2[i - 1]
        end = new_block_idxs2[i]
        blockevents = events[start:end, 2]
        block_sizes.append((blockevents, len(blockevents)))
        if len(blockevents) < min_block_size:
            rm_range = slice(start, end)
            rm_idxs.append(rm_range)

        if len(blockevents) > max_block_size:
            rm_range_strt = start + max_block_size
            rm_range_end = end
            rm_range = slice(rm_range_strt, rm_range_end)
            rm_idxs.append(rm_range)

    # remove them all at once
    keep_mask = np.ones_like(events, dtype=bool)
    for ix_pair in rm_idxs: keep_mask[ix_pair] = False
    keep_mask_use = keep_mask[:, 0]
    events_clean = events[keep_mask_use,]

    return events_clean


# code fede wrote to find events to delete/inject within a block
# I guess the injection parts needs to be able to see if a chunk of events
#  are missing from, e.g., the "start/stop" IDs that have events at the same time for the
#  matching "start instr/stop instr" events....
# But it might not be necessary after all, since altering the trig thresh
# cleaned the events in the first place.
# This was the case only with one file though..
def _check_events(events, sfreq, dist_thresh=10000):
    _mcp_event_id = {
        'inst_start/left': 10,  # 216 = 1101 1000b
        'start/left': 20,  # 208 = 1101 0000b
        'inst_stop/left': 30,  # 152 = 1001 1000b
        'stop/left': 40,  # 144 = 1001 0000b

        'inst_start/right': 50,  # 200 = 1100 1000b
        'start/right': 60,  # 192 = 1100 0000b
        'inst_stop/right': 70,  # 136 = 1000 1000b
        'stop/right': 80  # 128 = 1000 0000b
    }

    events_to_rm = []
    events_to_add = []
    # Need to use the sample numbers from the first "column" in the events array
    #  to find the indices to remove and then return that instead
    event_sample_nums_to_rm = []

    to_check = find_event_block_boundaries(events, dist_thresh=dist_thresh)

    for i in range(1, len(to_check)):
        start = to_check[i - 1]
        end = to_check[i]

        print('Checking block {i} in range '
              '[{start}, {end}]')

        block_events = events[start:end, :]

        for event_to_fix, event_id in _mcp_event_id.items():
            print(event_to_fix)
            id_mask = block_events[:, 2] == event_id
            t_events = block_events[id_mask]
            diffs = np.diff(t_events[:, 0]) / sfreq
            bad_idx = np.where(np.logical_or(diffs < 27, diffs > 30))[0]
            if len(bad_idx):
                print('Found bad {event_to_fix} events diff: ')
                for t_idx in bad_idx:
                    print('\tEvent {t_idx + 1} @ {t_events[t_idx][0]} => '
                          'diff is {diffs[t_idx]}')
                    should_remove = t_idx + 1 not in bad_idx
                    if should_remove:
                        print(
                            '\tShould remove {t_idx} @ {t_events[t_idx][0]}')
                        event_sample_nums_to_rm.append(t_events[t_idx][0])
        # Check that between the instruction trigger and the next we have 2.7s
        t_start = 0
        prev_inst = False
        for i_event, event in enumerate(events):
            if event[2] in [10, 30, 50, 70]:  # Inst, we count from here
                t_start = event[0]
                prev_inst = True
            else:
                diff = (event[0] - t_start) / sfreq
                if 2.6 > diff or diff > 2.8:
                    print('Event {i_event} @ {event[0]} does not have the '
                          'preceding instruction in range (diff = {diff})')
                if prev_inst is False:
                    prev_start = event[0] - round(2.71875 * sfreq)
                    prev_code = event[2] - 10
                    print('\t Should inject [{prev_start}, 0, {prev_code}]')
                    events_to_add.append([prev_start, 0, prev_code])
                prev_inst = False

        event_counts = Counter(block_events[:, 2])
        print('This block event counts:')
        for t_name, t_id in _mcp_event_id.items():
            print('\t{t_name} => {event_counts[t_id]}')
        print('=====================================\n')
    event_counts = Counter(events[:, 2])
    print('Overall Event counts:')
    for t_name, t_id in _mcp_event_id.items():
        print('\t{t_name} => {event_counts[t_id]}')

    return events_to_rm, events_to_add


def clean_event_id_pair(events, id_pair):
    id1, id2 = id_pair
    all_1 = np.where(events[:, 2] == id1)[0]
    all_2 = np.where(events[:, 2] == id2)[0]

    rm_ix = []

    if all_1[len(all_1)-1] == len(events)-1:
        # don't outright delete within this function; just return list of all indices to delete
        rm_ix.append(all_1[len(all_1)-1])
        all_1 = all_1[:-1]

    # this removes events where the next one is not the next ID in the matching pair
    # e.g., current ID is 80. If next ID is not 70, then remove the current ID.
    check_1 = [events[[all_1 + 1], 2] != id2][0][0]
    rm_1 = list(all_1[check_1])
    rm_ix.extend(rm_1)

    if len(all_2) != 0:
        check_2 = [events[[all_2-1],2] != id1][0][0]
        rm_2 = list(all_2[check_2])
        rm_ix.extend(rm_2)

    rm_unique = np.unique(rm_ix)

    return rm_unique


# same as below, but now use sub function on id pairs to find events to remove
def clean_events(events):
    event_counts = Counter(events[:, 2])
    event_ids = list(event_counts.keys())
    event_ids.sort()

    if len(event_ids) % 2 == 0:
        id_pairs = [(event_ids[i], event_ids[i+1]) for i in range(0, len(event_ids), 2)]
    else:
        print('Uneven amount of event IDs found!')
        return events

    out = []
    for pair in id_pairs:
        rm_ixs = clean_event_id_pair(events, id_pair=pair)
        out.append(rm_ixs)

    # concatenate all arrays together
    rm_ix_all = np.concatenate(out, axis=0).astype(int)
    # and remove
    events = np.delete(events, rm_ix_all, axis=0)

    return events


def clean_events_old_dont_use(events):
    # updated to extract event ids by itself
    event_counts = Counter(events[:, 2])
    event_ids = list(event_counts.keys())
    event_ids.sort()

    l_start, l_stop, r_start, r_stop = event_ids

    all_1 = np.where(events[:, 2] == l_start)[0]  # == 1)[0]
    all_2 = np.where(events[:, 2] == l_stop)[0]  # == 2)[0]
    all_3 = np.where(events[:, 2] == r_start)[0]  # == 3)[0]
    all_4 = np.where(events[:, 2] == r_stop)[0]  # == 4)[0]

    # if the last trigger is a 1 or a 3, need to not include it / remove it
    # otherwise can't check if the next one after is a 2 or 4 because there
    # is no trigger after it.
    if all_1[len(all_1)-1] == len(events)-1:
        events = np.delete(events, all_1[len(all_1)-1], axis=0)
        all_1 = all_1[:-1]
        
    check_1 = [events[[all_1+1],2] != l_stop][0][0]  # != 2][0][0]
    rm_1 = all_1[check_1]

    if len(all_3) != 0: 
        if all_3[len(all_3)-1] == len(events)-1:
            events = np.delete(events, all_3[len(all_3)-1], axis=0)
            all_3 = all_3[:-1]
            
    check_3 = [events[[all_3+1],2] != r_stop][0][0]  # != 4][0][0]
    rm_3 = all_3[check_3]

    # if all of the triggers after the '1' triggers are NOT '2',
    # then remove the one that has a False in this vector

    # the the trues from above to get indices of triggers to remove
    # do the opposite for 2 and 4
    # i.e., check if they all have a 1 or 3 before them
    # index to remove the 'list of list'
    if len(all_2) != 0:
        check_2 = [events[[all_2-1],2] != l_start][0][0]  # != 1][0][0]
        rm_2 = all_2[check_2]
        
    if len(all_4) != 0:
        check_4 = [events[[all_4-1],2] != r_start][0][0]  # != 3][0][0]
        rm_4 = all_4[check_4]
    # use the Trues in the above to get the indices from all_X
    # to know which triggers to remove

    # these are just used to determine which variables to concatenate together...
    # shouldn't affect the function itself.
    if ((len(all_2) != 0) & (len(all_1) != 0)) & ((len(all_3) == 0) & (len(all_4) == 0)):
        rm_all = np.concatenate((rm_1, rm_2), axis=None)
        
    elif ((len(all_2) == 0) & (len(all_1) == 0)) & ((len(all_3) != 0) & (len(all_4) != 0)):
        rm_all = np.concatenate((rm_3, rm_4), axis=None)
        
    else:
        rm_all = np.concatenate((rm_1, rm_3, rm_2, rm_4), axis=None)

    events = np.delete(events, rm_all, axis=0)

    return events


def clean_events2(evnts):
    event_counts = Counter(evnts[:, 2])
    event_ids = event_counts.keys()

    # get the most common count (ie, the first one in this sorted list)
    # then subset the events for the IDs with that number of counts
    most_common_events = event_counts.most_common()
    most_common_count_num = most_common_events[0][1]

    good_ids = [d for d in event_ids if event_counts[d] == most_common_count_num]
    # find ids to remove and remove them
    remove_ids = np.setdiff1d(list(event_ids), good_ids)
    # invert to pick the ones to keep
    rm_idx = np.invert(np.in1d(evnts[:, 2], remove_ids))
    new_evnts = evnts[rm_idx, :]

    # check to ensure number of unique events is now 4
    assert len(np.unique(new_evnts[:, 2])) == 4, "Number of unique event IDs is not 4!"

    return new_evnts


def insert_missing_chans(raw, missing, sfreq, ch_type='eeg'):
    fakeinfo = mne.create_info(ch_names=missing, sfreq=sfreq, ch_types=[ch_type] * len(missing))
    # for now, insert other channels' data as the fake data.
    fakedata = np.random.rand(len(missing), len(raw.times))
    # random data still produces all nan from interpolate_bads
    fakechans = mne.io.RawArray(fakedata, fakeinfo)

    raw.add_channels([fakechans], force_update_info=True)

    # reset montage after inserting new channels
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)

    return raw


def read_data(data, use_ch, tmin=0., tmax=10., fmin=.5, fmax=50.,
              n_epo_segments=1, ref_chans=None, hand_use=None,
              rename_chans=False, chan_dict=None, rename_chan_type=None, insert_missing=False,
              event_fl=None, is_control=False):
    """Parameters
    raw_fname : str
        file path of the raw.
    tmin : float
        epochs tmin
    tmax : float
        epochs tmax (each trial lasts for 10 second)
    fmin : float
        low-bound of bandpass filter.
    fmax : float
        high-bound of bandpass filter.
    n_epo_segments : int
        creates additional subevents in between each event. This allows
        creating multiple epochs for the very same 10-second-long trial,
        and thus increases the number of samples for the classifier.

        Be sure to use a LeaveGroup CV if n_epo_segments > 1, so as to
        ensure that the testing samples are not coming from trials
        that where used during training.
    """

    # Read raw data
    raw = mne.io.read_raw_fif(data, preload=True)
    # add in missing channels if specified
    if insert_missing:
        missing_chans = list(np.setdiff1d(use_ch, raw.info['ch_names']))
        if missing_chans:
            raw = insert_missing_chans(raw=raw, missing=missing_chans, sfreq=raw.info['sfreq'])
            raw.info['bads'] = missing_chans

    # rename channels if need to
    if rename_chans:
        if not chan_dict:
            sys.exit("Need to specify a channel dictionary if renaming channels")
        mne.rename_channels(raw.info, chan_dict)
        # now change channel type
        for i in range(len(raw.info['chs'])):
            if raw.info['chs'][i]['ch_name'] in chan_dict.values():
                raw.info['chs'][i]['kind'] = mne.channels.channels._human2fiff[rename_chan_type]

    picks = mne.pick_channels(raw.info['ch_names'], include=use_ch)
    if ref_chans is None:
        ref_chans = []
    mne.set_eeg_reference(raw, ref_channels=ref_chans, copy=False)

    sfreq = raw.info['sfreq']

    # Filter
    if (fmin is not None) and (fmax is not None):
        raw.filter(fmin, fmax)

    # Read events, and generate subevents
    # need to add a try statement here, since some of the control files
    # have their triggers in the 'Event' channel instead of the default.
    try:
        events = mne.find_events(raw)
    except ValueError:
        events = mne.find_events(raw, stim_channel=["Event"])

    if not list(events):
        print("File has no triggers")
        return None

    # Even number of trials to ensure class balance
    if any(trg in [5] for trg in events[:, 2]):
        events[:, 2] = events[:, 2] - 1

    # this more accurately removes extra events/triggers and should prevent the shift_events
    # function from breaking
    if (1 in events[:, 2] or 2 in events[:, 2]) and (3 in events[:, 2] or 4 in events[:, 2]):
        events = clean_events(events)

    # reset trigger from the onset of instruction to offset of instruction
    events = shift_events(events, sfreq, start_right=3, start_left=1, stop_right=4, stop_left=2, is_control=is_control)
    # make sure the events changed
    if events is None:
        return None

    # Remove events for a specific hand
    if hand_use == "left":
        print("using left hand")
        trgs = np.array([x[2] for x in events])
        keep_ix = np.logical_or(trgs == 1, trgs == 2)

        events = events[keep_ix]

    elif hand_use == "right":
        print("using right hand")
        trgs = np.array([x[2] for x in events])
        keep_ix = np.logical_or(trgs == 3, trgs == 4)

        events = events[keep_ix]

    new_events = list()
    for event in events:

        # Generate new events in case we want to subsegment each trial into
        # multiple epochs:
        for repeat in range(n_epo_segments):
            event_ = list(event)
            # evenly distribution across the 10 seconds
            event_[0] += repeat * raw.info['sfreq'] * 10. / n_epo_segments
            new_events.append(event_)
    events = np.array(new_events, int)

    # Add trial information
    # # identify unique trial (useful in case of epoch splitting)
    trial_id = np.cumsum(np.diff(np.r_[events[0, 2], events[:, 2]]) != 0)
    metadata = DataFrame(dict(id=events[:, 2], trial=trial_id))
    # # give a unique column for movement versus rest
    metadata['move'] = False
    metadata.loc[metadata.query('id in (1, 3)').index, 'move'] = True

    # Segment data
    if insert_missing:
        reject_bads = False
    else:
        reject_bads = True

    epochs = mne.Epochs(raw, tmin=tmin, tmax=tmax,
                        events=events, metadata=metadata,
                        picks=picks, proj=False,
                        baseline=None, preload=True,
                        reject_by_annotation=reject_bads)
    return epochs


# this is used to read/aggregate the data from the fifs and separate event files that have the cleaned
# events from the fedebox.
def read_data_fedebox(data, event_fl, use_ch, tmin=0., tmax=10., fmin=.5, fmax=50.,
                      n_epo_segments=1, ref_chans=None, hand_use=None,
                      rename_chans=False, chan_dict=None, rename_chan_type=None, insert_missing=False, is_control=False):
    """Parameters
    raw_fname : str
        file path of the raw.
    tmin : float
        epochs tmin
    tmax : float
        epochs tmax (each trial lasts for 10 second)
    fmin : float
        low-bound of bandpass filter.
    fmax : float
        high-bound of bandpass filter.
    n_epo_segments : int
        creates additional subevents in between each event. This allows
        creating multiple epochs for the very same 10-second-long trial,
        and thus increases the number of samples for the classifier.

        Be sure to use a LeaveGroup CV if n_epo_segments > 1, so as to
        ensure that the testing samples are not coming from trials
        that where used during training.
    """

    # Read raw data
    raw = mne.io.read_raw_fif(data, preload=True)

    if insert_missing:
        missing_chans = list(np.setdiff1d(use_ch, raw.info['ch_names']))
        if missing_chans:
            raw = insert_missing_chans(raw=raw, missing=missing_chans, sfreq=raw.info['sfreq'])
            raw.info['bads'] = missing_chans

    # rename channels if need to
    if rename_chans:
        if not chan_dict:
            sys.exit("Need to specify a channel dictionary if renaming channels")
        mne.rename_channels(raw.info, chan_dict)
        # now change channel type
        for i in range(len(raw.info['chs'])):
            if raw.info['chs'][i]['ch_name'] in chan_dict.values():
                raw.info['chs'][i]['kind'] = mne.channels.channels._human2fiff[rename_chan_type]

    picks = mne.pick_types(raw.info, eeg=False, stim=False, eog=False,
                           ecg=False, misc=False, include=use_ch)
    if ref_chans is None:
        ref_chans = []
    mne.set_eeg_reference(raw, ref_channels=ref_chans, copy=False)

    # Filter
    if (fmin is not None) and (fmax is not None):
        raw.filter(fmin, fmax)

    # Read events, and generate subevents
    try:
        events = mne.read_events(event_fl)
    except AttributeError:
        print("File has no triggers")
        return None

    if np.all(np.unique(events[:, 2]) == np.arange(10, 90, 10)):
        use_ids = [20, 40, 60, 80]
        l_start, l_stop, r_start, r_stop = use_ids

    # case for old files
    elif np.all(np.in1d(np.unique(events[:, 2]), np.arange(1, 5, 1))):
        use_ids = [1, 2, 3, 4]
        l_start, l_stop, r_start, r_stop = use_ids

    # if hit this condition, then this is presumably a new file with messed up events.
    # skip it for now.
    else:
        print('File has weird events. Skipping...')
        return None

    # filter for specific events to use (defaults to 20, 40, 60, 80; aka start/stop move after instruction ends
    event_idx = np.where([x in use_ids for x in events[:, 2]])[0]
    if len(event_idx) != 0:
        events = np.take(a=events, indices=event_idx, axis=0)

    # Remove events for a specific hand if specified
    if hand_use == "left":
        print("using left hand")
        trgs = np.array([x[2] for x in events])

        keep_ix = np.logical_or(trgs == l_start, trgs == l_stop)
        events = events[keep_ix]

    elif hand_use == "right":
        print("using right hand")
        trgs = np.array([x[2] for x in events])

        keep_ix = np.logical_or(trgs == r_start, trgs == r_stop)
        events = events[keep_ix]

    new_events = list()
    for event in events:
        # Generate new events in case we want to subsegment each trial into multiple epochs:
        for repeat in range(n_epo_segments):
            event_ = list(event)
            # evenly distribution across the 10 seconds
            event_[0] += repeat * raw.info['sfreq'] * 10. / n_epo_segments
            new_events.append(event_)
    events = np.array(new_events, int)

    # Add trial information
    # # identify unique trial (useful in case of epoch splitting)
    trial_id = np.cumsum(np.diff(np.r_[events[0, 2], events[:, 2]]) != 0)
    metadata = DataFrame(dict(id=events[:, 2], trial=trial_id))
    # # give a unique column for movement versus rest
    metadata['move'] = False
    query_check = 'id in (' + str(l_start) + ', ' + str(r_start) + ')'
    metadata.loc[metadata.query(query_check).index, 'move'] = True

    # Segment data
    epochs = mne.Epochs(raw, tmin=tmin, tmax=tmax,
                        events=events, metadata=metadata,
                        picks=picks, proj=False,
                        baseline=None, preload=True)
    return epochs


def fix_epochs(epochs, good_len=10):
    if np.all(np.unique(epochs.events[:, 2]) == np.arange(20, 90, 20)):
        use_ids = [20, 40, 60, 80]
        l_start, l_stop, r_start, r_stop = use_ids

    # case if using the start of the spoken command as the events, but I don't think this will ever be the case
    elif np.all(np.unique(epochs.events[:, 2]) == np.arange(10, 80, 20)):
        use_ids = [10, 30, 50, 70]
        l_start, l_stop, r_start, r_stop = use_ids

    # case for old files
    elif np.all(np.unique(epochs.events[:, 2]) == np.arange(1, 5, 1)):
        use_ids = [1, 2, 3, 4]
        l_start, l_stop, r_start, r_stop = use_ids

    # make sure the event ids are all INTS before continuing.
    r_start, r_stop, l_start, l_stop = int(r_start), int(r_stop), int(l_start), int(l_stop)
    md = copy.deepcopy(epochs.metadata)
    # make indentifier column to indicate where start/stop block pairs change
    md['blockchange'] = np.nan
    # initialize first row
    md['blockchange'][0] = 1

    md['id_shift'] = md['id'].shift()
    counter = 1
    for ix, rough in md.iterrows():
        # if on the first row and the id_shift if blank, just insert the first 'counter' number
        if np.isnan(rough['id_shift']):
            md.loc[ix, 'blockchange'] = counter
        if rough['id'] == l_start and rough['id_shift'] == l_stop:
            # then, whenever one of these cases happens, increase the counter by 1 and set it as the current
            # blockchange value
            counter += 1
            md.loc[ix, 'blockchange'] = counter
        elif rough['id'] == r_start and rough['id_shift'] == l_stop:
            counter += 1
            md.loc[ix, 'blockchange'] = counter
        elif rough['id'] == r_stop and rough['id_shift'] == l_start:
            counter += 1
            md.loc[ix, 'blockchange'] = counter
        elif rough['id'] == r_start and rough['id_shift'] == r_stop:
            counter += 1
            md.loc[ix, 'blockchange'] = counter
        elif rough['id'] == l_start and rough['id_shift'] == r_stop:
            counter += 1
            md.loc[ix, 'blockchange'] = counter
        else:
            # if block doesn't change, then keep previous start/stop block pair number
            md.loc[ix, 'blockchange'] = counter

    # now find the counts of each blockpair. If they don't equal the good_len, remove them.
    cnt = Counter()
    for blockchange in md['blockchange']:
        cnt[blockchange] += 1

    # find indices of blocks without their other start/stop pair
    bad_blocks = [k for k, v in cnt.items() if v != good_len]
    bad_blocks_ix = md.loc[md['blockchange'].isin(bad_blocks)].index.tolist()

    # if any were found, remove them
    if bad_blocks_ix:
        epochs.drop(bad_blocks_ix, reason='USER')

    return epochs


# main analysis function
def run_pipeline(epochs, fmin=8.0, fmax=30.0,
                 pipe_type="tangent", overlap=0.9, delays=[1, 2, 4, 8],
                 bands=((1, 3), (4, 7), (8, 13), (14, 30)),
                 n_permutations=500, max_iter=1000,
                 laplac_ref=True):

    X = epochs.get_data()
    y = epochs.metadata['move']  # decode move vs rest (not rest/left/right)

    if pipe_type == "cosp":
        print('running cosp')

        sfreq = epochs.info['sfreq']

        cosp = make_pipeline(
            CospCovariances(fmin=fmin, overlap=overlap, fmax=fmax, fs=sfreq),
            CospBoostingClassifier(make_pipeline(TangentSpace('logeuclid'),
                                                 LogisticRegression('l2'))))
        cv = LeaveOneGroupOut()
        groups = np.array(epochs.metadata['trial']/2., int)
        scores = cross_val_score(cosp, X, y, scoring='roc_auc',
                                 cv=cv, groups=groups)
        # also calculate standard error of mean
        se = scores.std(0) / np.sqrt(len(scores))

        score = scores.mean(0)
        permutation_scores = []
        for i in range(n_permutations):
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)
            permutation_scores.append(cross_val_score(cosp, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)
        print("cosp AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores # data,

    elif pipe_type == "tangent":
        print('running tangent')

        ts_log = make_pipeline(Covariances(estimator='oas'),
                               TangentSpace('logeuclid'),
                               LogisticRegression('l2'))

        cv = LeaveOneGroupOut()
        groups = np.array(epochs.metadata['trial']/2., int)
        scores = cross_val_score(ts_log, X, y, scoring='roc_auc',
                                 cv=cv, groups=groups)

        se = scores.std(0) / np.sqrt(len(scores))

        score = scores.mean(0)
        permutation_scores = []
        for i in range(n_permutations):
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)
            permutation_scores.append(cross_val_score(ts_log, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("tangent space AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores  # data,

    elif pipe_type == "riemann":
        print('running riemann')

        riemman_log = make_pipeline(Covariances(estimator='oas'),
                                    TangentSpace('logeuclid'),
                                    LogisticRegression('l2'))

        cv = LeaveOneGroupOut()
        groups = np.array(epochs.metadata['trial']/2., int)
        scores = cross_val_score(riemman_log, X, y, scoring='roc_auc',
                                 cv=cv, groups=groups)

        se = scores.std(0) / np.sqrt(len(scores))

        score = scores.mean(0)
        permutation_scores = []
        for i in range(n_permutations):
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)
            permutation_scores.append(cross_val_score(riemman_log, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("Riemann CSP AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores  # data,

    elif pipe_type == "hankel":
        print('running hankel')

        hankel_csp_log = make_pipeline(HankelCovariances(delays=delays, estimator='oas'),
                                       TangentSpace('logeuclid'),
                                       LogisticRegression('l2'))

        cv = LeaveOneGroupOut()
        groups = np.array(epochs.metadata['trial']/2., int)
        scores = cross_val_score(hankel_csp_log, X, y, scoring='roc_auc',
                                 cv=cv, groups=groups)

        se = scores.std(0) / np.sqrt(len(scores))

        score = scores.mean(0)
        permutation_scores = []
        for i in range(n_permutations):
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)
            permutation_scores.append(cross_val_score(hankel_csp_log, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("Hankel CSP AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores

    elif pipe_type == "psd":
        # Fiilter: 1. to 30. Hz
        print('running psd')

        epochs.info['description'] = 'standard/1020'
        if laplac_ref is True:
            epochs = pycsd.epochs_compute_csd(epochs)  # compute Curent Source Densisty

        # Compute power spectral densities for each frequency band
        psd_data, frequencies = psd_multitaper(epochs, fmin=fmin, fmax=fmax)

        n_epochs, n_chans, n_freqs = psd_data.shape

        # Setup X and y using the different frequency bands
        X = np.zeros((n_epochs, n_chans, len(bands)))
        for ii, (fmin, fmax) in enumerate(bands):
            # find frequencies
            freq_index = np.where(np.logical_and(frequencies >= fmin, frequencies <= fmax))[0]
            # mean across frequencies
            X[:, :, ii] = psd_data[:, :, freq_index].mean(2)

        # Vectorize across frequency bands
        X = X.reshape(n_epochs, -1)

        clf = make_pipeline(StandardScaler(), LinearSVC(max_iter=max_iter))

        cv = LeaveOneGroupOut()
        groups = np.array(epochs.metadata['trial']/2., int)
        scores = cross_val_score(clf, X, y, scoring='roc_auc',
                                 cv=cv, groups=groups)

        se = scores.std(0) / np.sqrt(len(scores))

        score = scores.mean(0)
        permutation_scores = []
        for i in range(n_permutations):
            sys.stdout.flush()
            print('running permutation ' + str(i+1))
            order = list(range(len(y)))
            np.random.shuffle(order)
            permutation_scores.append(cross_val_score(clf, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("PSD AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))

        # return epochs to in order to count how many files have incomplete blocks
        return score, se, pvalue, permutation_scores

    else:
        raise NameError("Incorrect pipeline name specified")
