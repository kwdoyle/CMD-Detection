#!/usr/bin/env python3

import ast
import pandas as pd
from mne.stats import fdr_correction as fdr

pd.options.mode.chained_assignment = None

# NOTE: can get the model output dataframe w/ cmd significance Y/N in a column per recording
# from read_file2's output

def find_signif_files2(x, orig_dat):
    new_df = pd.DataFrame()
    for ix, rw in x.iterrows():
        signifs = rw['out'][0]
        tmp_dat = orig_dat[orig_dat['mrn'] == rw['mrn']]
        # can I just add the new pvalues and signifs to the table?
        new_p = rw['out'][1]
        tmp_dat['p_correct'] = new_p
        tmp_dat['p_signif'] = signifs

        new_df = new_df.append(tmp_dat)

    return new_df


def read_file2(input_file):
    model_data = pd.read_csv(input_file)
    # wow, wtf, I didn't remove CS and MCSp recordings in this?????
    # not explicitly necessary with newer analyzed files as I don't put them in separate folders
    # per CRS-R score group.
    model_data = model_data[~model_data['rec_name'].str.contains('/CS/')]
    model_data = model_data[~model_data['rec_name'].str.contains('/MCSp/')]
    # remove rows where model failed to run (AUC == NaN or == 0)
    bad_recs = model_data[pd.isnull(model_data.AUC) | model_data.AUC == 0]
    model_data.drop(bad_recs.index, inplace=True)
    bad_rec_names = bad_recs.rec_name.tolist()

    model_data['pvalue'] = pd.to_numeric(model_data['pvalue'])
    # extract MRNs
    try:
        model_data['mrn'] = model_data['rec_name'].str.split('_').apply(lambda x: x[0]).astype('int')
    except ValueError:
        # print('Getting MRNs from full file path instead..')
        model_data['mrn'] = model_data['rec_name'].str.split('/').apply(lambda x: x[2]).str.split('_').apply(
            lambda x: x[0]).astype('int')
    # FDR correct to obtain significant files
    model_data_grp = model_data.groupby('mrn')
    fdr_pvals = pd.DataFrame(model_data_grp.apply(lambda x: fdr(x.pvalue))).reset_index()
    fdr_pvals = fdr_pvals.rename(columns={"mrn": "mrn", 0: "out"})
    # find each significant file.
    # Can get the number of signif patients from this
    new_model_data = find_signif_files2(fdr_pvals, orig_dat=model_data)
    # sort by mrn
    new_model_data = new_model_data.sort_values(by=['mrn'])

    return new_model_data, bad_rec_names


def get_cmd_recs(df):
    cmd_rec_dat = df[df.p_signif == True]
    noncmd_rec_dat = df[df.p_signif == False]

    cmd_dat_splt = cmd_rec_dat.rec_name.str.split('/').tolist()
    noncmd_dat_splt = noncmd_rec_dat.rec_name.str.split('/').tolist()

    cmd_recs = [x[len(x) - 1] for x in cmd_dat_splt]
    noncmd_recs = [x[len(x) - 1] for x in noncmd_dat_splt]

    return cmd_recs, noncmd_recs


def parse_perm_auc(df):
    perm_aucs = {}
    for ix, rw in df.iterrows():
        rec = rw.rec_name
        # get rid of extra filepath if it exists
        if len(rec.split('/')) > 1:
            rec = rec.split('/')[len(rec.split('/')) - 1]

        permaucs = ast.literal_eval(rw.perm_scores)
        perm_aucs[rec] = permaucs

    return perm_aucs


def main(fl):
    cmd_data_fl = fl
    # read model outputs to find cmd and non-cmd recordings
    cmd_data, bad_recs = read_file2(cmd_data_fl)
    cmd_recs, noncmd_recs = get_cmd_recs(cmd_data)
    # get permuated aucs per recording
    perm_aucs = parse_perm_auc(cmd_data)

    return cmd_recs, noncmd_recs, bad_recs, perm_aucs


if __name__ == '__main__':
    main()
