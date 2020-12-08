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

    # 9/5/19 UPDATE: It appears that Natus NOW uses that middle column of 'Duration' for a given clip note name,
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
    # UPDATE: since I'm now just reading all ACUTAL 3 COLUMNS from line 5 onward above,
    # can now just start at the start of 'table'
    for col in range(0, len(table)):
        table[col][0] = table[col][0].strip()

    # get start time
    start = datetime.strptime(table[1][0], '%H:%M:%S')

    # turn timestamps into seconds from start of eeg clip
    for col in range(1, len(table)):
        tmstmp = datetime.strptime(table[col][0], '%H:%M:%S')
        table[col][0] = (tmstmp - start).total_seconds()

    # 9/5/19 UPDATE: if table does indeed have 3 columns, remove the middle one ('duration')
    if table.shape[1] == 3:
        # no idea why 'obj=1' works to delete the entire column.
        table = np.delete(table, obj=1, axis=1)

    return table


# ==============================================================================
# ## Function to get rows with 'left' and 'right' events from event table
# ==============================================================================

def get_event_rows(event_tab, sfreq):
    labels = []
    # start at 5th row, where all events should always start,
    # to avoid including names with 'right' in them (e.g., 'Wright')
    for row in event_tab:  # [5:]:
        # at first, I got around having the useless events by also searching for the word 'hand'
        # but then realized I could just check if 'montage' isn't in the line instead
        # UPDATE: Nope, 'hand' needs to be checked for too, since I just found some text files that talk about
        # 'right cheek twitching', and they were being included as events.
        # and 'montage' not in str(row[1]).lower()    and 'montage' not in str(row[1]).lower()
        #  and 'montage' not in str(row[1]).lower()    and 'montage' not in str(row[1]).lower()
        # UPDATE: This shouldn't be checking the second columns of the table, since NOW the second column
        # could have info in it and can't be stripped out from transform_timestamps.
        # Now the column with the labels is column 3, but REALLY it's just always the last column.
        # Make this instead check row[len(row)-1]
        if (
                ('right' in str(row[len(row) - 1]).lower() and 'hand' in str(row[len(row) - 1]).lower()) or
                ('open' in str(row[len(row) - 1]).lower() and 'right' in str(row[len(row) - 1]).lower()) or
                # I think the events demarked w/ an 'r' or 'l' need to be checked w/ a space after the letter,
                # otherwise ANY word w/ an 'r' or 'l' and has 'open' in it will be counted.
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

# So this change does allow for the partially croped out trigger block at the end of some of the files,
# but in doing so, it also allows for any small trigger blocks of the same size that could be strewen
# throughout the file.
# maybe can add something that will only keep them if they're at the end of the file?
# I think that would entail a whole restructuring of how this function works, though..

def remove_bad_trigs(chan, sec, smallsec, sfreq, min_block_size=16):
    # This removes triggers that aren't in a block of 16 (8 each stop and start).
    # If we don't want that, we'll have to edit this.
    # I think this can be done by just changing the number that 'count' is checked against.
    triggers = np.nonzero(chan[0])

#    sec = max seconds allowed between any good triggers
#    smallsec = max seconds allowed where any trigger occurring closer than this to another is removed

    indices = []
    bad_trigs = []
    # set to stupidly high number to start, so that the conditional check won't "fail" when it checks
    # like it did when this was set to 'None'
    distbehnd = 9999999999999
    for i in range(0, len(triggers[0])):
        if i+1 < len(triggers[0]):
            # print "current trigger" + str(triggers[0][i])
            # calculate distance from current value & next value
            dist = triggers[0][i+1] - triggers[0][i]
            # if distance between values is an acceptable amount
            if dist < sec * sfreq:
                # just add this value to a running list of values
                # which make up the current column
                indices.append(triggers[0][i])

                # print "appending trigger " + str(triggers[0][i]) + " ;" +  str(dist) + " < " + str(sec * sfreq)

            # if distance between values is greater than an acceptable amount
            if dist > sec * sfreq:
                # add this current value to the running list
                indices.append(triggers[0][i])
                # and see how long the list is
                count = len(indices)

                # print "appending trigger " + str(triggers[0][i]) + " ;" + str(dist) + " > " + str(sec * sfreq)
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
                    # print "count is >= 16 so they're all good"

            # check the behind distances so that, for any triggers that are too close together,
            # the last one in that group can still be checked
            if i != 0:
                distbehnd = triggers[0][i] - triggers[0][i-1]

            # if distance between values is an acceptable amount
            # and also greater than a stupidly small amount (there are some triggers that are way too close together)
            if dist < sec * sfreq and dist < smallsec * sfreq or distbehnd < sec * sfreq and distbehnd < smallsec * sfreq:
                # checking for the last one in a column w/ too small dist between
                # don't add to to indices; just delete?
                # print str(triggers[0][i]) + " is too close to the last one, so it's being removed"
                # print "dist of trigger " + str(triggers[0][i]) + " and one in front, " + str(triggers[0][i+1]) + " is " + str(dist)
                # print "dist of trigger " + str(triggers[0][i]) + " and one behind, " + str(triggers[0][i-1]) + " is " + str(distbehnd)

                bad_trigs.append(triggers[0][i])
                chan[0][triggers[0][i]] = 0
                # print "set this one to 0: " + str(triggers[0][i])

        # case for last value
        if i == len(triggers[0])-1:
            # since this is the final value, I have to check the one behind it instead
            dist = triggers[0][i] - triggers[0][i-1]
            # if the distance is larger than acceptable, then set it to 0
            if dist > sec * sfreq:
                # print str(triggers[0][i]) + " has too large a distance as the last value, so removing it"

                bad_trigs.append(triggers[0][i])
                chan[0][triggers[0][i]] = 0
                # print "set this one to 0: " + str(triggers[0][i])

            # also need to see if the current column I'm in at the end is < 16 triggers.
            # it's possible that there is a partial column of triggers at the very end

            # if it's not larger than acceptable, check column length
            if dist < sec * sfreq:
                # print "distance is acceptable for the last value. have to check column length"

                indices.append(triggers[0][i])
                count = len(indices)
                if count < min_block_size:
                    # print "count was less than 16, so these are removed: "
                    # print indices
                    for j in indices:
                        bad_trigs.append(j)
                        chan[0][j] = 0
                        # print "set this one to 0: " + str(j)

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
    # I guess set a condition that, if the value == 0, don't overwrite it
    lastline = []   # this is the previous line from the current line
    maxthresh = thresh[len(thresh)-1]['range'][1]
    for sample in trigs:
        for line in thresh:
            # account for overlaps in thresholds first
            if lastline:
                if sample > line['range'][0] and sample < line['range'][1] and \
                        sample > lastline['range'][0] and sample < lastline['range'][1]:  # and chan[0][sample] != 0:
                    # it's not re-checking the removed tennis trigger because, it first gets removed when
                    # it's found to be only in the tennis event, and then isn't being reassigned as right
                    # because it's now set to 0 and isn't being checked again
                    # if I find later that some actual bad triggers that were removed by remove_bad_trigs
                    # are now being re-assigned because of this,
                    # then maybe I should set bad triggers to something other than 0.
                    # print 'sample in both'
                    # print sample
                    # print line#['range']
                    # print lastline#['range']
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
            # I guess leave it just in case I want to make the 'left' values something other than 1.
            # Apparently this part is necessary now that I am checking for overlaps in thresholds.
                elif sample > line['range'][0] and sample < line['range'][1] and chan[0][sample] != 0:
                    # print 'sample in the current threshold'
                    # print sample
                    # print line#['range']
                    if 'left' in str(line['label']).lower() or 'l' in str(line['label']).lower():
                        # print 'assigned left'
                        chan[0][sample] = start_left  # pass
                    if 'right' in str(line['label']).lower() or 'r' in str(line['label']).lower():
                        # print 'assigned right'
                        chan[0][sample] = start_right
                    # remove alice and tennis
                    if 'alice' in str(line['label']).lower():
                        chan[0][sample] = 0
                    # I have to do something with the boolean stuff here to correctly remove the tennis triggers and
                    # leave in any good, overlapping triggers but I cannot figure it out right now.
                    # I have to check that, for the current trigger, it falls in the 'tennis' thresh line, but
                    # DOESN'T ALSO fall in a 'right' or 'left' line... THAT'S the problem. (right now the below line
                    # won't remove anything, because the lastline is
                    # always going to either be a left or right hand line.)
                    # UPDATE: um.. I guess I did fix this..? I've never noticed any problems...
                    # if ('tennis' in str(line['label']).lower() and 'right' not in str(lastline['label']).lower()) and
                    # ('tennis' in str(line['label']).lower() and 'left' not in str(lastline['label']).lower()):
                    if 'tennis' in str(line['label']).lower():
                        # print 'removed tennis'
                        chan[0][sample] = 0

            # if lastline is empty, then do this:
            else:
                if sample > line['range'][0] and sample < line['range'][1] and chan[0][sample] != 0:
                    # print "sample in the current threshold (and lastline hasn't been defined yet)"
                    # print sample
                    # print line#['range']
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
            # print lastline

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
    #        print str(trigs[i]) + ": curr trig"
    #        print str(chan[0][trigs[i]]) + ": curr trig val"
    #        print str(trigs[i+1]) + ": next trig"
    #        print str(chan[0][trigs[i+1]]) + ": next trig val"
    #        print str(check_front) + ": check front"
    #        print str(t_range_for_trig * sfreq) + ": max val for range"
    #        print check_front in range(0, int(t_range_for_trig * sfreq))
    #        print "\n"
            if check_front in range(0, int(t_range * sfreq)):
                #print trigs[i]
                if chan[0][trigs[i+1]] == start_left:
    #                print "it's 1"
    #                print '\n'
                    chan[0][trigs[i+1]] = stop_left

                if chan[0][trigs[i+1]] == start_right:
    #                print "it's 3"
    #                print '\n'
                    chan[0][trigs[i+1]] = stop_right

    return chan


# ==============================================================================
# ## Make new threshold for removing bad triggers
# ==============================================================================

def make_new_thresh(chan, sfreq, tbuff, start_right=3, start_left=1, stop_right=4, stop_left=2, sndpass=False):
    # need to generate new threshold ranges for triggers, as some text files might include two blocks in one event..
    # maybe find the distanes between all triggers, and then define a group as when the distance jumps
    # to a value larger than the current running average distance..? and when a jump occurs, don't include that distance in the running average.
    # (but doing it this way could be bad too, since there are two files that have no break between any trigger groups. maybe it doesn't matter?)
    # (maybe, if finding groups this way fails, then default to using the thresholds determined by the text file).
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
                # So, for all cases except one, this works. it fails when a new block starts out with the
                # wrong hand trigger and then jumps to extra triggers for the stop of the correct hand.
                # Need it to basically do the curr_start_trig check also only if I guess the length of the block
                # is more than 1? 2? some small number.
                # Maybe check like how long in seconds the time is between the firstofblock and endofblock.
                # If it's smaller than an acceptable amount, then this isn't a new block.                                             # I think this number here is supposed to be the length of an event block in seconds. If I set it to 4 minutes, it seems to work for one edf file at least
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
#    dists_in_block = []
    for i in range(1, len(trigs)):

        curr_trig = trigs[i]  # chan[0][trigs[i]]
        prev_trig = trigs[i-1]  # chan[0][trigs[i-1]]

#        curr_dist = (curr_trig - prev_trig) / sfreq
#        dists_in_block.append(curr_dist)

        if i == 1:
            previous_triggers.append([chan[0][trigs[i-1]], trigs[i-1]])
            previous_triggers.append([chan[0][trigs[i]], trigs[i]])
            continue

        # always append both the trigger and its index together
        # this has to go after the check for len of previous_triggers, otherwise
        # this list has the next trigger (potentially of the next block) in it
        # previous_triggers.append( [chan[0][trigs[i]], trigs[i]] )

        # calculate average distance between triggers in the current block
        # Although I'd have to use the average once the block is determined,
        # but I can't determine the end of the block without the average.
#        mean_of_block_dists = np.max(dists_in_block)
        # Might want to edit this again so that it doesn't jump to a new block
        # if the current block happens to have MORE than 16 triggers

        # I'm probably going to have to calculate btwn_block_time as the average distance between the triggers
        # in the current block. If it's greater than that, then this is a new block.

        # mean_of_block_dists  # 50 was just an arbitrary amount of seconds that the time btwn triggers should be larger
        # than if this is the start of a new block. now it's a set parameter
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
# ###### ...make a new function to fix misassigned trigs,
# ###### but have it use new_thresh because it should be easier
# I think I need both clean_missassigned_trigs functions because the first one
# works without the assumption of clearly defined trigger blocks.
# After all is said and done, this one can be used to clean up any remaining ones.
# ==============================================================================

def clean_missassigned_trigs2(chan, new_thresh, start_right=3, start_left=1, stop_right=4, stop_left=2):
    for i in range(0, len(new_thresh)):
        trgs = chan[0][new_thresh[i][0]:new_thresh[i][1]+1]
        # trgs_u = np.unique(trgs, return_counts=True)
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
                # start counting trigs from the end of this block
#                idx_of_key = np.where(chan[0] == key)[0]
#                idx_of_idx = np.where(np.logical_and(idx_of_key >= int(line[0]), idx_of_key <= int(line[1])))
                # or just use the current 'line' in new_thresh? since that's the whole interval for this block
                # these are the values in the current block
#                curr_block = chan[0][line[0]:line[1]]
#                for i, e in reversed(list(enumerate(curr_block))):
#                    print i, e
                # find what the triggers in this block, excluding 0
                trigs_in_block = list(filter(lambda x: x != 0, countdic.keys() ))

                # need to index entirety of chan.
                # Plan is to count backwards from the number of triggers in this block
                # (assuming the last triggers are assigned to the correct hand),
                # then remove any left after counting 16 of them.
                all_idx_trig = [i for i, x in enumerate(chan[0]) if x in trigs_in_block] # == 3 or x == 4]

                # aa[line[0]:line[1]]
                # np.where(np.logical_and(idx_of_key >= int(line[0]), idx_of_key <= int(line[1])))
                # and then filter that between the start and end indices of this block
                idx_of_all_idx = np.where(np.logical_and(all_idx_trig >= line[0], all_idx_trig <= line[1]))
                # then use these indices to get the indices from 'aa' which are the indices from chan
                # that relate to this block. oh my god.
                # aa[t[0]]
                # this works?
                block_idx_trig = [all_idx_trig[i] for i in idx_of_all_idx[0]]

                # now go through each index in reverse and count how many of each trigger there are.
                # once the current trigger is the same as the previous one,
                # remove all remaining triggers and then set the current trigger as a start event.
                # (not sure how robust this method will be)
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
                        # set all trigs "after" these (really they're before,
                        # but since the index order is reversed, they're "after") to 0
                        # print chan[0][rev_as_lst[i+1:]]
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
                                # print chan[0][rev_as_lst[i]]
                                chan[0][rev_as_lst[i]] = start_right

                            if event_w_most_trigs == start_left or event_w_most_trigs == stop_left:
                                chan[0][rev_as_lst[i]] = start_left

                        break

    return chan
#                    print rev_as_lst[i]
#                    print curr_trig, prev_trig


#                count = 0
#                for i in reversed(block_idx_trig):
#                    if count == 0:
#                        prev_trig = chan[0][i]
#                        count += 1
#
#                    curr_trig = chan[0][i]
#                    count += 1
#                    if curr_trig == prev_trig:
#                        print i
#                        print curr_trig, prev_trig
                         # set all trigs "after" this one to 0


# can now reject outliers from the sri using the below function

def reject_outliers(data, m=2.):
    d = np.abs(data - np.median(data))
    mdev = np.mean(d)
    s = d/mdev if mdev else 0.
    return data[s < m]


# ==============================================================================
# ## shift events function
# ==============================================================================

def shift_events(events, sfreq, start_right=4, start_left=2, stop_right=5, stop_left=3):

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

    # subtract the n-1th of kmR from the nth of smR
    # Right hand
    kri = []
    for i in range(0, len(smR_np)):
        kri.append((smR_np[i] - kmR_np[i]) - (10 * sfreq))
        # this works for kri.
    kri_med = round(np.median(kri))

    # it = iter(kmR_np)
    sri = []
    for i in range(1, len(kmR_np)):
        sri.append((kmR_np[i] - smR_np[i-1]) - (15 * sfreq))
        # this includes the huge gap that might be present between different runs of R to L...
    # get rid of outlier
    sri_med = round(np.median(reject_outliers(np.array(sri))))

    # Left hand
    kli = []
    for i in range(0, len(smL_np)):
        kli.append((smL_np[i] - kmL_np[i]) - (10 * sfreq))
    kli_med = round(np.median(kli))

    sli = []
    for i in range(1, len(kmL_np)):
        sli.append((kmL_np[i] - smL_np[i-1]) - (15 * sfreq))
    sli_med = round(np.median(reject_outliers(np.array(sli))))

    # just return all these values for now
    # return kri_avg, sri_avg, kli_avg, sli_avg
    # test editing events by converting to pandas df and then back to np array

    if np.isnan(kri_med) is not True:
        newkmR = kmR + np.int64(kri_med)
        newsmR = smR + np.int64(sri_med)
        events_df.loc[events_df['trig'] == start_right, 'samp'] = newkmR
        events_df.loc[events_df['trig'] == stop_right, 'samp'] = newsmR

    if np.isnan(kli_med) is not True:
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
#        if -40.0 in kri_list:  ### I guess the larger sample distances are maybe from a different language?
#            raise ValueError('bad kri calculation in ' + fname1)
        [sri_list.append(x) for x in sri - np.int64(round(np.mean(reject_outliers(np.array(sri)))))]
    if np.isnan(np.mean(kli)) is not True:
        [kli_list.append(x) for x in kli - np.int64(round(np.mean(kli)))]
        [sli_list.append(x) for x in sli - np.int64(round(np.mean(reject_outliers(np.array(sli)))))]


# old method for clean_trigger_blocks:
#        for key in countdic.keys():
#            if key == 0:
#                continue
#            if countdic[key] > 8: #!= 8:
#                ######### Ok, so I should have this take the total size of the current block into account
#                ##### Right now, it's only seeing extra triggers if there are more than 8 of them.
#                ##### But there might be a block which has less than 8 and still has a double trigger.
#                ##### I guess maybe if the number of each start/stop don't match, then the one that has a higher number has the extra trigger.
#                ## This might be removing extra ones even if they're normal..? but no, cause then there should be an extra stop/start which is removed too..
#                print 'trigger label ' + str(key) + ' has ' + str(countdic[key]) + ' triggers instead of 8 in range ' + str(line)
#
#                # I guess always remove the second extra trigger?
#                # check how many extra ones there are.
#                # if it's 1, remove 1. if 2, remove 2.. etc,
#                extra_trigs = countdic[key] - 8
#
#                #np.where(chan[0] == key)[int(line[0]-1):int(line[1]+1)]
#
#                # first need to find indices of where chan == this key,
#                # THEN find which of these indices are equal to/in the current thresh range / 'line'
#                # and then use the second one as the index in chan to set to 0.
#                # I think it makes sense to remove the second one, because the patient will have already
#                # started moving their R/L hand from this earlier trigger, and will presumably keep moving it
#                # once they hear the same command again, so might as well just remove that second one.
#                # For the stop moving ones, can remove the second extra trigger for the same reason:
#                # they will already have stopped moving their hand after the first command.
#
#                ##### Oh, but maybe we should remove the first one? because then the response to the audio cue of the second stop trigger
#                ##### will show up during the first?
#
#                #### Also, this is only considering a second extra trigger. If there's 2 extra, it only removes one of them.
#                #### Maybe have it compare how many there are to 8 and remove the difference
#                #### e.g., there's 10 triggers; should be 8; 10-8=2, so need to remove 2 extra triggers.
#                #### if there are 9 triggers, then remove 9-8=1 extra trigger (like how it normally works)
#
#                idx_of_key = np.where(chan[0] == key)[0]
#                idx_of_idx = np.where(np.logical_and(idx_of_key >= int(line[0]), idx_of_key <= int(line[1])))
#                # this then gets the index of all of chan which refers to the second extra trigger in this current event block.
#                # gotta add in some if extra_trigs == 1, then do the below. if it's 2, then remove that trig and the next one too.
#                #chan_idx_to_rm = idx_of_key[idx_of_idx][1]
#                ## so this looks confusing, but it is just getting the indices of which triggers (out of all of them)
#                ## to remove by always starting at the 2nd extra trigger (index 1) to the nth extra trigger
#                ## (where the index is however many extra triggers there are)
#                range_to_index = range(1, extra_trigs+1)
#
#                chan_idx_to_rm = idx_of_key[idx_of_idx][np.ix_(range_to_index)]  #[1,extra_trigs]
#
#                # remove the trigger
#                chan[0][chan_idx_to_rm] = 0
#                # set 'success' to 1
#                #success = 1
#
#
#
#            # also add a check so that, if a count for one of the key-types is small
#            # (basically, just not the majority of what the other triggers are)
#            # i.e., it's the wrong hand trigger amongst the current hand trigger group,
#            # then remove all of those triggers.
#            # I think this mostly happens when the wrong hand-event is played instead of the correct one.
#            # e.g., one patient has the correct number of start and stop, but the first start is for the R hand instead of the L hand.
#            # --but wait, that can't be why, since the R/L determination is done here based on the text file. I guess it was just a coincidence?
#            if countdic[key] <= 4:  # maybe re-define this 'better' at some point? Right now set anything less than 4 triggers as bad and should all be removed?
#                print 'trigger label ' + str(key) + ' has ' + str(countdic[key]) + ' triggers instead of 8 in range ' + str(line)
#                idx_of_key = np.where(chan[0] == key)[0]
#                idx_of_idx = np.where(np.logical_and(idx_of_key >= int(line[0]), idx_of_key <= int(line[1])))
#                chan_idx_to_rm = idx_of_key[idx_of_idx]  # don't index the second one from this, since I want to remove all of them.
#
#                chan[0][chan_idx_to_rm] = 0
#                #success = 1
#
#
#
#    # This is one last checkthrough to catch any remaining triggers.
#    # The triggers that would be caught by this part occur when there are still 8 of that specific trigger,
#    # but it happens to be a stop trigger and it the block starts with it.
#    new_trgs_ix = np.where(chan[0] != 0)
#    for i in range(1, len(new_trgs_ix[0])):
#        cur_val = chan[0][new_trgs_ix][i]
#        prev_val = chan[0][new_trgs_ix][i-1]
#
#        if i == 1:
#            if (prev_val == 2 or prev_val == 4):
#                chan[0][new_trgs_ix[0][i-1]] = 0
#
#            continue
#
#
#        if cur_val == 1:
#            if (prev_val != 2 and prev_val != 4) and prev_val != 0:
#                chan[0][new_trgs_ix[0][i]] = 0
#
#        if cur_val == 2:
#            if prev_val != 1 and prev_val != 0:
#                chan[0][new_trgs_ix[0][i]] = 0
#
#        if cur_val == 3:
#            if (prev_val != 4 and prev_val != 2) and prev_val != 0:
#                chan[0][new_trgs_ix[0][i]] = 0
#
#        if cur_val == 4:
#            if prev_val != 3 and prev_val != 0:
#                chan[0][new_trgs_ix[0][i]] = 0
#
#
#
#
#    return chan #, success
#


# ==============================================================================
# ## Function to remove amplitudes greater than a specified value. Defaults to 120uV
# ==============================================================================
# if 0 < n_comp < 1, then it is the cumulative percentage of explained variance used to choose components.
def remove_artifacts(raw, picks, n_comp=0.95, thresh=240e-6):
    ica = mne.preprocessing.ICA(n_components=n_comp, method='fastica')
    # is this NOT removing things that are above this rejection threshold?
    # i.e., doing the opposite of what I want?
    # I don't think so..? since it only states 'Artifact detected in ...' when the rejection parameter is defined
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

            for k in range(0, len(new_pick)):  # len(close_chans_bads)):
                # need to get indices of these close chans in raw
                # Close_chans indices:

                # I think this should be:
                # new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

                # new_pick = raw.info['ch_names'].index(close_chans_bads[k])

                # Need to check if sd is > 90% of more than 2 of its neighbors.
                # If it is, and this occurrs in more than 3 windows, then mark chan as bad.
                # print "neighbor sd"
                # print raw._data[new_pick, start:stop].std()
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
                    # at end checking all 4 neighbors, see if count is >=2. if it is, then channel may be bad.
                    # store this channel in 'potential bads' or something.
                    # then continue doing this for all windows.
                    # at the very end, see the running list of how many times a channel was added to 'potential bads'
                    # if it was more than 3 (2?) times, then the channel is definitely bad.

            # Do same thing to find flat channels, but use all 6 nearest neighbors

            # should make new_pick out here and then iterate over those
            new_pick2 = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for x in range(0, len(new_pick2)):  # len(close_chans_flat)):
                # new_pick2 = raw.info['ch_names'].index(close_chans_flat[x])
                # check if < 20% of at least 3 of 6 neighbors
                val = raw._data[new_pick2[x], start:stop].std() * per_flat

                if val > curr_chan_sd:
                    count_flat += 1
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
                # print 'checking for potential bad chans'

                # need to get indices of these close chans in raw
                # Close_chans indices:

                # I think this should be:
                # new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

                # new_pick = raw.info['ch_names'].index(close_chans_bads[k])

                # Need to check if sd is > 90% of more than 2 of its neighbors.
                # If it is, and this occurrs in more than 3 windows, then mark chan as bad.
                # print "neighbor sd"
                # print raw._data[new_pick, start:stop].std()
                val = raw._data[new_pick[k], start:stop].std() * per_bad
                # print 'neighb chan sd+70% for bad: ' + str(val)
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
                    # at end checking all 4 neighbors, see if count is >=2. if it is, then channel may be bad.
                    # store this channel in 'potential bads' or something.
                    # then continue doing this for all windows.
                    # at the very end, see the running list of how many times a channel was added to 'potential bads'
                    # if it was more than 3 (2?) times, then the channel is definitely bad.

            # Do same thing to find flat channels, but use all 6 nearest neighbors

            # should make new_pick out here and then iterate over those
            new_pick2 = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for x in range(0, len(new_pick2)):  # len(close_chans_flat)):
                # print 'checking for potential flat chans'

                # new_pick2 = raw.info['ch_names'].index(close_chans_flat[x])
                # check if < 20% of at least 3 of 6 neighbors
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


# I think I need to check for flat channels first, and then when checking for bad channels,
# if a neighboring channel if flat, don't include it in the checking-comparison.
# do something like:
    # if all_names[j] is a flat channel, then instead use all_names[j+1] to compare
# Yeah I think I need to not include the flat chans in the comparisons.
# I'm pretty sure this is what HP did in his, since he uses 'setdiff(neighbors, flat chans)
# to subtract out the flat chans from the neighbor chans.


# I guess I should make find_bad_chans and find_flat_chans two different functions.
# Or at least two seperate loops in the same function.


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
        # (this is using the indices that correspond to the channels of interest in the raw data)
        # right now it's printing the max/sd for each channel for each window.
        # I need it to then compare these values to each channel's neighbors still for each window
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

            # should make new_pick out here and then iterate over those
            # I guess this way didn't matter here since the indices would have been the same
            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_bads)

            for k in range(0, len(new_pick)):  # len(close_chans_bads)):
                # print 'checking for potential bad chans'

                # need to get indices of these close chans in raw
                # Close_chans indices:

                # I think this should be:
                # new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

                # new_pick = raw.info['ch_names'].index(close_chans_bads[k])

                # Need to check if sd is > 90% of more than 2 of its neighbors.
                # If it is, and this occurrs in more than 3 windows, then mark chan as bad.
                # print "neighbor sd"
                # print raw._data[new_pick, start:stop].std()
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
                    # at end checking all 4 neighbors, see if count is >=2. if it is, then channel may be bad.
                    # store this channel in 'potential bads' or something.
                    # then continue doing this for all windows.
                    # at the very end, see the running list of how many times a channel was added to 'potential bads'
                    # if it was more than 3 (2?) times, then the channel is definitely bad.

            # Do same thing to find flat channels, but use all 6 nearest neighbors

            # should make new_pick out here and then iterate over those
            new_pick2 = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for x in range(0, len(new_pick2)):  # len(close_chans_flat)):
                # print 'checking for potential flat chans'

                # new_pick2 = raw.info['ch_names'].index(close_chans_flat[x])
                # check if < 20% of at least 3 of 6 neighbors
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
                    if count_flat > len(close_chans_flat) / 2:
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
            # print "curr chan: " + all_names[j]
            # find the values for the current window
            # print start, stop
            # can use 'all_names[j]' as the key for the nearest dictionary
            close_chans_flat = nearest[all_names[j]]  # take all 6 nearest neighbors
            # get index of current chan
            # 70% more than current channel's sd:
            curr_chan_sd = raw._data[j, start:stop].std() #* 1.7
            # print 'curr chan sd: ' + str(curr_chan_sd)
            # make a counter of how many times it was greater
            count_flat = 0
            # should make new_pick out here and then iterate over those
            # I guess this way didn't matter here since the indices would have been the same
            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_flat)

            for k in range(0, len(new_pick)):  # len(close_chans_bads)):

                val = raw._data[new_pick[k], start:stop].std() * per_flat
                # print 'neighb chan sd+70% for bad: ' + str(val)
                # val corresponds to the neighbor-channel.
                # if current channel is larger than its 70%-increased-neighbors 2 or more times, then it's probably bad
                # print curr_chan_sd, val
                if val > curr_chan_sd:
                    count_flat += 1
                    # print close_chans_flat[k]
                    # print '40% of neighb chan is greater: ' + str(val)
                if k == len(close_chans_flat)-1:
                    # print count_flat
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
            # print start, stop
            # can use 'all_names[j]' as the key for the nearest dictionary
            close_chans_bad = nearest[all_names[j]][0:4]  # take 4 neighbors
            # get index of current chan
            # 70% more than current channel's sd:
            curr_chan_sd = raw._data[j, start:stop].std() # * 1.7
            # print 'curr chan sd: ' + str(curr_chan_sd)
            # make a counter of how many times it was greater
            count_bad = 0
            # should make new_pick out here and then iterate over those
            # I guess this way didn't matter here since the indices would have been the same
            # add in any flat channels here in 'exclude'

            # this technically isn't comparing against 4 channels each time, though, if a flat channel is excluded
            # but is that necessarily bad? should a further channel even be used in place of a flat neighbor channel?
            new_pick = mne.pick_channels(raw.info['ch_names'], close_chans_bad, exclude=flat_chans)

            for k in range(0, len(new_pick)):

                val = raw._data[new_pick[k], start:stop].std() * per_bad
                # print 'neighb chan sd+70% for bad: ' + str(val)
                # val corresponds to the neighbor-channel.
                # if current channel is larger than its 70%-increased-neighbors 2 or more times, then it's probably bad
                # print curr_chan_sd, val
                if val < curr_chan_sd:
                    count_bad += 1
                    # print close_chans_bad[k]
                    # print 'neighb chan sd+70% is less: ' + str(val)
                if k == len(close_chans_bad)-1:
                    # print count_bad
                    # checking if was greater than half of the number of 4th nearest channels (i.e., 2)
                    # don't do greater or equal to
                    if count_bad > len(close_chans_bad) / 2:  # maybe these do need to be greater or equal.
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

# ## ...wait, I don't need to modify the start triggers. it's only the stop ones.
# ==============================================================================
# L hand remains at 1 (default value)
# R hand = 3
def assign_start_mod_for_baseln(trigs, thresh, chan, start_right=3, start_left=1):
    # This will overwrite any bad trigs that were set to 0 to 3 if they technically fall in the event range,
    # even if they aren't legit triggers.
    # I guess set a condition that, if the value == 0, don't overwrite it
    lastline = []   # this is the previous line from the current line
    maxthresh = thresh[len(thresh)-1]['range'][1]
    for sample in trigs:
        for line in thresh:
            # account for overlaps in thresholds first
            if lastline != []:
                if sample > line['range'][0] and sample < line['range'][1] and sample > lastline['range'][0] and \
                        sample < lastline['range'][1] and chan[0][sample] != 0:
                    # print 'sample in both'
                    # print line['range']
                    # print lastline['range']
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
            # print lastline

            # turn 'alice' events to 0 ####
            # this is redundant when checking for bad trigs and setting them to 0
            # if 'alice' in str(line['label']).lower():
            #   chan[0][sample] = 0

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
    # t_range is the time allowed between a start/stop event. This will vary for some patients,
    # possibly due to English v. Spanish spoken events taking different amounts of time to say
    for i in range(0, len(trigs)):
        if i+1 < len(trigs):
            # this is essentially the same as checking for bad triggers, except now
            # it just checks if the value is 1 or 3 and sets the value to 2 or 4 instead
            check_front = trigs[i+1] - trigs[i]
            if check_front in range(0, int(t_range * sfreq)):
                # print trigs[i]
                if chan[0][trigs[i+1]] == start_left:
                    chan[0][trigs[i+1]] = stop_left

                if chan[0][trigs[i+1]] == start_right:
                    chan[0][trigs[i+1]] = stop_right

    return chan


def clean_trigger_blocks2(events, dist_thresh=10000, block_size_thresh=12):
    event_counts = Counter(events[:, 2])
    event_ids = list(event_counts.keys())
    event_ids.sort()
    l_start, l_stop, r_start, r_stop = event_ids

    labs = events[:, 2]
    # index differences
    diffs = np.diff(events[:, 0])
    # gaps where block changes based on distance
    # add 1 to the indices to get the start of the next block and not the end of the previous one
    new_block_dist = np.where(diffs > dist_thresh)[0] + 1

    # gaps where block changes based on id
    l_change = []
    r_change = []
    for i in range(1, len(labs)):
        if (labs[i - 1] not in (l_start, l_stop)) and (labs[i] in (l_start, l_stop)):
            l_change.append(i)
        if (labs[i - 1] not in (r_start, r_stop)) and (labs[i] in (r_start, r_stop)):
            r_change.append(i)

    new_block_id = np.array(l_change + r_change)
    new_block_id.sort()

    new_block_idxs = np.unique(np.concatenate((new_block_dist, new_block_id), axis=None))
    # add in the start and end!!!!!!!
    new_block_idxs2 = np.concatenate((new_block_idxs, (0, len(labs) - 1)))
    new_block_idxs2.sort()

    # count number of event ids between the new block idx
    # and pick out blocks that are too small and should be removed
    rm_idxs = []
    block_sizes = []
    for i in range(1, len(new_block_idxs2)):
        start = new_block_idxs2[i - 1]
        end = new_block_idxs2[i]
        blockevents = events[start:end, 2]
        block_sizes.append((blockevents, len(blockevents)))
        if len(blockevents) < block_size_thresh:
            rm_range = slice(start, end)
            rm_idxs.append(rm_range)

    # remove them all at once
    keep_mask = np.ones_like(events, dtype=bool)
    for ix_pair in rm_idxs: keep_mask[ix_pair] = False
    # o m f g. need to use only the 1st 'column' to index with otherwise I lose my dimensions of the events
    keep_mask_use = keep_mask[:, 0]
    events_clean = events[keep_mask_use,]

    return events_clean


# Will remove extraneous triggers from all the events
# i.e., remove instruction from an incomplete trial
# e.g., a 'keep moving' without the following 'stop moving'
# TODO omg... this should idealy use some internal function that does whatever this function does
#  but for a given event id instead of trying to do it for L/R start/stop at once
#  e.g., this works when the BROKEN events are passed to it because there's only 4 unique IDs
#  but when the DC7 channel works and the events are correctly-parsed, this fails because there's
#  more than 4 unique IDs.

# TODO test doing the above on recording6 I guess to ensure it works on a file I already know it worked for before
#  and then try on recording5 again.

# TODO or... not?
#  actually, this could work if it just did it on "pairs"
#  in the case w/ 4 unique IDs, there's 2 pairs. L start/stop and R start/stop.
#  in the case w/ 8 unique IDs, there should be 4 pairs. L start/stop, R start/stop, L instr start/stop, R instr start/stop.
def clean_event_id_pair(events, id_pair):
    id1, id2 = id_pair
    all_1 = np.where(events[:, 2] == id1)[0]
    all_2 = np.where(events[:, 2] == id2)[0]

    rm_ix = []

    if all_1[len(all_1)-1] == len(events)-1:
        # don't outright delete within this function; just return list of all indices to delete
        # events = np.delete(events, all_1[len(all_1)-1], axis=0)
        rm_ix.append(all_1[len(all_1)-1])
        all_1 = all_1[:-1]

    check_1 = [events[[all_1 + 1], 2] != id2][0][0]
    rm_1 = list(all_1[check_1])
    rm_ix.extend(rm_1)




def clean_events(events):
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
    # IS no trigger after it.
    # !!! I think if these are true, the trigger should be outright removed--
    # not just remove the index
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
    # do stupid indexing to remove the 'list of list' thing that numpy does
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
    # np.where(evnts[:, 2] == remove_ids)
    # invert to pick the ones to keep
    rm_idx = np.invert(np.in1d(evnts[:, 2], remove_ids))
    new_evnts = evnts[rm_idx, :]

    # check to ensure number of unique events is now 4
    assert len(np.unique(new_evnts[:, 2])) == 4, "Number of unique event IDs is not 4!"

    return new_evnts


def insert_missing_chans(raw, missing, sfreq, ch_type='eeg'):
    fakeinfo = mne.create_info(ch_names=missing, sfreq=sfreq, ch_types=[ch_type] * len(missing))  # [ch_type] * len(missing))
    # for now, insert other channels' data as the fake data.
    # fakedata = np.zeros((len(missing), len(raw.times)))
    fakedata = np.random.rand(len(missing), len(raw.times))
    # fakedata = raw._data[:len(missing), :]
    # random data still produces all nan from interpolate_bads
    # fakedata = np.random.rand(len(missing), len(raw.times))
    fakechans = mne.io.RawArray(fakedata, fakeinfo)

    raw.add_channels([fakechans], force_update_info=True)

    # reset montage after inserting new channels
    montage = mne.channels.read_montage('standard_1020')
    raw.set_montage(montage)

    return raw


def read_data(data, use_ch, tmin=0., tmax=10., fmin=.5, fmax=50.,
              n_epo_segments=1, ref_chans=None, hand_use=None,
              rename_chans=False, chan_dict=None, insert_missing=False):
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

    # ??? why was pick_types being used instead of the simpler pick_channels???
    # I don't **think** this will break anything by switching
    # picks = mne.pick_types(raw.info, eeg=False, stim=False, eog=False,
    #                        ecg=False, misc=False, include=use_ch)
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
    # This needs to go BEFORE the shift_events function
    # because if uneven amounts of start and stop are given to this function,
    # it tries to subtract values of lists from different lengths
    # and returns an error.
    # ..I have no idea WHY this wasn't returning an error in earlier runs with the same data.
    # if events.shape[0] % 2 != 0:
    #    events = events[:-1]

    # make sure events aren't set as 2,3; 4,5 and if they are, subtract 1 from all
    # UPDATE: OK, since 5 is the only trigger that SHOULDN'T be included if they really are set to 2,3; 4,5
    # I should just check if 5 is in there or not. Otherwise I can run into a problem if a file only has,
    # e.g., right hand triggers, because then this comes out as true and it subtracts 1, causing the events
    # to make no sense (they'd be stop left, start right, stop left, etc.)
    if any(trg in [5] for trg in events[:, 2]):
        events[:, 2] = events[:, 2] - 1

    # this more accurately removes extra events/triggers and should prevent the shift_events
    # function from breaking
    if (1 in events[:, 2] or 2 in events[:, 2]) and (3 in events[:, 2] or 4 in events[:, 2]):
        events = clean_events(events)

    # old_events = events.copy()
    # reset trigger from the onset of instruction to offset of instruction
    events = shift_events(events, sfreq, start_right=3, start_left=1, stop_right=4, stop_left=2)
    # make sure the events changed
    # assert (np.all(old_events == events) == False)

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


# TODO this is just a copy of what's found in preproc.
#  can update it with whatever version fede fixes in preproc at some point
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


# This is used with the eeg data obtained using the new Fedeboxes. The events used are different, as well as
# the find_events function needing the consecutive=True argument
def read_data2(data, use_ch, tmin=0., tmax=10., fmin=.5, fmax=50.,
               n_epo_segments=1, ref_chans=None, hand_use=None,
               rename_chans=False, chan_dict=None):
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
    # rename channels if need to
    if rename_chans:
        if not chan_dict:
            sys.exit("Need to specify a channel dictionary if renaming channels")
        mne.rename_channels(raw.info, chan_dict)

    picks = mne.pick_types(raw.info, eeg=False, stim=False, eog=False,
                           ecg=False, misc=False, include=use_ch)
    if ref_chans is None:
        ref_chans = []
    mne.set_eeg_reference(raw, ref_channels=ref_chans, copy=False)

    # Filter
    if (fmin is not None) and (fmax is not None):
        raw.filter(fmin, fmax)

    # Read events, and generate subevents
    # need to add a try statement here, since some of the control files
    # have their triggers in the 'Event' channel instead of the default.
    try:
        events = mne.find_events(raw, consecutive=True)
    except ValueError:
        events = mne.find_events(raw, consecutive=True, stim_channel=["Event"])

    if not list(events):
        print("File has no triggers")
        return None

    # TODO do this only if event ids are NOT 10 20 30 40, 50 60 70 80
    #  and, if they are, then use the events r_start=60, r_stop=80, l_start=20, l_stop=40
    #  I guess ignore all of this if triggers are 10 20 30 40 50 60 70 80?
    # so stupid. make range end at 90 with step 10 so that 80 is the last value.
    if np.all(np.unique(events[:, 2]) == np.arange(10, 90, 10)):
        use_ids = [20, 40, 60, 80]
        l_start, l_stop, r_start, r_stop = use_ids

    # TODO I haven't actually tested this on old files yet...
    # case for old files
    elif np.all(np.unique(events[:, 2]) == np.arange(1, 5, 1)):
        use_ids = [1, 2, 3, 4]
        l_start, l_stop, r_start, r_stop = use_ids

    # if hit this condition, then this is presumably a new file with messed up events.
    # skip it for now.
    else:
        print('File has weird events. Skipping...')
        return None

    # filter for specific events to use (defaults to 20, 40, 60, 80; aka start/stop move after instruction ends
    # [from new process_triggers function fede wrote to process triggers from fedebox])
    event_idx = np.where([x in use_ids for x in events[:, 2]])[0]
    # Need to check if event_idx is empty (I think this happens for the older files b/c none are 20, 40, 60, 80)
    # if it is, then this causes events to become empty too
    # althought I guess this really isn't needed if I'm defining the use_ids based off of
    # the events themselves
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


def fix_epochs(epochs, good_len=10, r_start=60, r_stop=80, l_start=20, l_stop=40):
    # make sure the event ids are all INTS before contunuing.
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


# run_pipeline runs on epochs now, not raw data
def run_pipeline(epochs, fmin=8.0, fmax=30.0,
                 pipe_type="tangent", overlap=0.9, delays=[1, 2, 4, 8],
                 bands=((1, 3), (4, 7), (8, 13), (14, 30)),
                 n_permutations=500, max_iter=1000,
                 laplac_ref=True):

    X = epochs.get_data()
    y = epochs.metadata['move']  # decode move vs rest (not rest/left/right)

    if pipe_type == "cosp":
        # sys.stdout.flush()
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
            # sys.stdout.flush()
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)  # marche
            permutation_scores.append(cross_val_score(cosp, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)
        # sys.stdout.flush()
        print("cosp AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores # data,

    elif pipe_type == "tangent":
        # sys.stdout.flush()
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
            # sys.stdout.flush()
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)  # marche
            permutation_scores.append(cross_val_score(ts_log, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("tangent space AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores  # data,

    elif pipe_type == "riemann":
        # sys.stdout.flush()
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
            # sys.stdout.flush()
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)  # marche
            permutation_scores.append(cross_val_score(riemman_log, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("Riemann CSP AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores  # data,

    elif pipe_type == "hankel":
        # sys.stdout.flush()
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
            # sys.stdout.flush()
            print('running permutation ' + str(i+1))
            order = range(len(y))
            np.random.shuffle(order)  # marche
            permutation_scores.append(cross_val_score(hankel_csp_log, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("Hankel CSP AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))
        return score, se, pvalue, permutation_scores

    elif pipe_type == "psd":
        # Fiilter: 1. to 30. Hz
        print('running psd')

        epochs.info['description'] = 'standard/1020'  # TODO: fix this depth electrode montage issue
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
            np.random.shuffle(order)  # marche
            permutation_scores.append(cross_val_score(clf, X, y[order], scoring='roc_auc',
                                      cv=cv, groups=groups[order]).mean(0))

        pvalue = (len(list(filter(lambda x: x >= score, permutation_scores))) + 1.0) / (n_permutations + 1)

        print("PSD AUC = %.2f +/-%.2f (p-value = %.3f )" % (score, se, pvalue))

        # return epochs to in order to count how many files have incomplete blocks
        return score, se, pvalue, permutation_scores

    else:
        raise NameError("Incorrect pipeline name specified")
