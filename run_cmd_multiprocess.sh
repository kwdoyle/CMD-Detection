#!/bin/bash

# Pass a directory with the fif files to analyze and the directory with the event files to this script
# and the cmd_detection_multiprocessing  should take care of the rest

FIFS=$1
EVENTS_DIR=$2
FIFS+="*.fif"  # this works somehow. it just auto-fills in all fif files using the wildcard.
NB_FILE_PER_JOB=1 # 5  # number should normally be 5. I GUESS you could do 1 file per job/core though
SAVEPATH='./model_outfiles/'
#EVENTS_DIR='../event_files/'  # specify this manually instead
OUTPUT_FL='psd_out_all.csv'
CONTROL='False'

# loop over all files until the NB_FILES_PER_JOB is reached
### I changed the h_vmem from 10G to 3G because I don't think each job uses 10G, and this might allow more to run at once
/home/kd2630/code/CMD-Detection/cmd_detection_multiprocessing.py \
      	--write_dir $SAVEPATH \
      	--events_dir $EVENTS_DIR \
	--n_per_job $NB_FILE_PER_JOB \
       	--rawfiles $FIFS \
 	--is_control $CONTROL \
	--combined_output_fl $OUTPUT_FL

