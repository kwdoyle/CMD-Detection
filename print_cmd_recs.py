#!/usr/bin/env python3

import sys
import parse_cmd as cmd


if __name__ == '__main__':
    model_output_file = sys.argv[1]
    cmd_recs, noncmd_recs, bad_recs, perm_aucs = cmd.main(fl=model_output_file)
    cmd_mrns = [x.split('_')[0] for x in cmd_recs]
    noncmd_mrns = [x.split('_')[0] for x in noncmd_recs]
    # clean just in case. wrapping with set removes any duplicates.
    cmd_mrns = list(set([x.replace('-', '') for x in cmd_mrns]))
    cmd_mrns.sort()
    noncmd_mrns = list(set([x.replace('-', '') for x in noncmd_mrns]))
    noncmd_mrns.sort()

    just_badrecs = [x.split('/')[-1] for x in bad_recs]
    badpats = list(set([x.split('_')[0] for x in just_badrecs]))
    badpats.sort()

    # somehow putting an astericks in front of the list prints them as individual values
    print('CMD Recordings:')
    print(*cmd_recs, sep='\n')
    print('\n')
    print('CMD Patients:')
    print(*cmd_mrns, sep='\n')
    print('\n')
    print('Non CMD Patients:')
    print(*noncmd_mrns, sep='\n')
    print('\n')
    print('Bad Recordings:')
    print(*bad_recs, sep='\n')
    print('\n')
    print('Patients with Bad Recordings:')
    print(*badpats, sep='\n')
    print('\n')
