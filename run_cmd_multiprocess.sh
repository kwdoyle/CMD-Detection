#!/bin/bash

# Pass a directory with the fif files to analyze and the directory with the event files to this script
# and the cmd_detection_multiprocessing  should take care of the rest

FIFS=$1
EVENTS_DIR=$2
FIFS+="*.fif"  # this works somehow. it just auto-fills in all fif files using the wildcard.
NB_FILE_PER_JOB='20' # 5  # number should normally be 5.
# cannot figure out how to use an int that ends with 0 without bash removing that 0 when passing to the script below.
# get around this by passing it as a string, then converting to int within the script.
SAVEPATH='./model_outfiles/'
OUTPUT_FL='psd_out_all.csv'
CONTROL='False'
# This is mainly for testing
#NPERM=0 #500

/home/kd2630/code/CMD-Detection/cmd_detection_multiprocessing.py \
      	--write_dir $SAVEPATH \
      	--events_dir $EVENTS_DIR \
	--n_per_job $NB_FILE_PER_JOB \
#	--nperm $NPERM \
       	--rawfiles $FIFS \
 	--is_control $CONTROL \
	--combined_output_fl $OUTPUT_FL

