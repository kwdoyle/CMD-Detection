#!/usr/bin/env bash

## NOTE this script must be run via: source ./set_files.sh [file_dir_path]
## otherwise it won't actually set the filelist variable

export filelist=$(cat files_to_analyze.txt)
