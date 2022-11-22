# CMD Detection Analysis
Detect Cognative Motor Dissociation (CMD) in patients through EEG analysis.

This is the set of core scripts used for the analysis in the paper
_Detection of Brain Activation in Unresponsive Patients with Acute Brain Injury._

## Pipeline:
rename_edfs.py | preproc.py | cmd_detection.py     
Note: any additional help on these files can be obtained by running them with the `--help` argument.     
e.g., `preproc.py --help`

#### Rename Files:

To run this analysis, your EEG data files should be in the .EDF file format.  
Unprocessed files, at least exported from a Natus database, have a name structure of:    
'LastName~ FirstName_[a long string of characters].edf'        

If you wish to first standardize the file names for your data, first run `rename_edfs.py` in the directory containing the EDF files 
either by specifying the directory with the `--cwd` argument or first switching to that directory.      
This file will then move all the recordings into a subdirectory named "Converted" with their new names.       
It should be noted that this step is not strictly necessary to run the analysis, but it does allow you to keep track of the eeg recordings much easier.

For a general renaming, the `--save_noname` argument can be specified as `True` and this will save the file with
whatever name is present before the first underscore, followed by the date of the recording obtained internally from the file.

If you wish to substitute the patient names with a different identifier value (e.g., an MRN), you can pass an excel file containing 'mrn' and 'name' columns containing the corresponding matching values per patient using the `--file` argument.     
e.g., `--file ./path/to/rename/excel.xlsx`               

#### Preprocessing:

To extract the event information from the eegs to then be used in the subsequent analysis step, `preproc.py` should next be run from the directory
containing the edf files to analyze.       
For each edf, this step will save the events, the edfs with a new '.fif' extention, and plots of the events in new subdirectories named 'event_files', 'fif_files', and 'event_plots', respectively.          

If event triggers are saved into a channel named 'Event' or 'Trigger Event' with information specifying whether the trigger is a 'start' or 'stop' move event saved into a text file (e.g., if exported from the notes section of the eeg in Natus),          
then the `--force_old_method` argument can be set to `True`.        
Otherwise it is assumed triggers, along with the information about them, are stored within the 'DC' channels.          
The general case should have the start/stop left/right hand events saved to DC5 through DC8.       
If any event is saved in a different channel, it can be remapped using the `--chan_dict` argument

#### Analysis:

For running the CMD detection, the file `cmd_detection_multiprocessing.py` should now be used.     
This is to take advantage of any and all CPU cores the computer you are running the analysis on may have.         
This can be done via the `run_cmd_multiprocess.sh` shell script, with the first argument passed be the directory containing the fif files
and the second argument being the directory conatining the event files.         
`./run_cmd_multiprocessing.sh ./fif_files/ ./event_files/`.           

Within `run_cmd_multiprocessing.sh`, the number of files to be run per-core can be specified.

The output files from the CMD detection are then saved in a new directory named 'model_outfiles/'

