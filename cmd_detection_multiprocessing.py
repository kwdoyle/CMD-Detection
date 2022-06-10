#!/usr/bin/env python3
import os
import argparse
from copy import deepcopy
import multiprocessing as mp
from cmd_detection import main as runcmd


CLI = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args, pool):
    flsuse = []
    allfiles = args.rawfiles
#    print("all files passed are:")
#    print(allfiles)
    jobs = []
    numjob = 0
#    args_all = []
    for i in range(len(allfiles)):
        flsuse.append(allfiles[i])
        if len(flsuse) == int(args.n_per_job):
            argsuse = deepcopy(args) 
            # ..I guess it's ok to overwrite the main args.rawfiles. it's a little janky but whatevever
            #args.rawfiles = flsuse 
            argsuse.rawfiles = flsuse 
            # Update the jobnumb to prevent any possible issues of different
            # jobs writing to the same file
            argsuse.num_job = str(numjob)
            jobs.append(pool.apply_async(runcmd, args=(argsuse, )))
            # then reset flsuse
            flsuse = []
            numjob += 1

        elif i == len(allfiles)-1: # base case for end if number of files is
        #not divisible by n_per_job
            argsuse = deepcopy(args) 
            argsuse.rawfiles = flsuse 
            argsuse.num_job = str(numjob)
            jobs.append(pool.apply_async(runcmd, args=(argsuse, ))) 
            # then reset flsuse
            flsuse = []
            numjob += 1

    # actually do need to save all jobs in a list and "submit" them all like this for them to run all together
    ok = [job.get() for job in jobs]



CLI.add_argument(
    "--cwd",
    type=str,
    default='.',
    help='the working directory to run this script from'
)

CLI.add_argument(
    "--write_dir",
    type=str,
    default='./model_outfiles/',
    help='directory to save the model output to. directory will be created if it does not exist'
)

CLI.add_argument(
    "--events_dir",
    type=str,
    default='./event_files/',
    help='directory containing the event files corresponding to each fif file'
)

CLI.add_argument(
    "--num_job",
    type=str,
    default='0',
    help='the current job number. this is automatically set when running this script via the cluster'
)

CLI.add_argument(
    "--rawfiles",
    nargs="*",
    type=str,
    default=None,
    help='the list of fif files to analyze. this is usually set using set_files.sh, '
         'but file paths can be also passed manually here'
)

CLI.add_argument(
    "--combined_output_fl",
    type=str,
    default='psd_out_all.csv',
    help='file that contains all the previous model output. '
         'recordings in this file that were previously analyzed will be skipped'
)

# this argument is mainly to reduce number of permutations when testing
CLI.add_argument(
    "--nperm",
    type=int,
    default=500,
    help='the number of permutations to run when checking the model AUC significance'
)

CLI.add_argument(
    "--n_per_job",
    #type=int,
    #default=5,
    type=str,
    default='5',
    help='the number of files to process per core'
)


CLI.add_argument(
    "--is_control",
    type=str2bool,
    nargs='?',
    const=True,
    default=False,
    help='specify if analyzing a control file or not. mainly due to controls only containing one hand. '
         'normally, a file is skipped if it does not contain both hands'
)


if __name__ == "__main__":
    args = CLI.parse_args()
    n_cores = os.cpu_count()
    pool = mp.Pool(n_cores)

    main(args, pool)

