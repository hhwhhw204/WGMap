import argparse
import sys

import numpy as np

np.random.seed(1)
import os
import math
import re
from subprocess import check_output
import pandas as pd
from os.path import exists

root_path = '../../../../'
def fea_mut_all(df_ori, str_col):
    df = df_ori.copy()
    types = ['Intron', 'IGR', 'RNA', 'Missense_Mutation', "3'UTR", 'lincRNA', "5'Flank", 'Silent', "5'UTR",
             'Splice_Site',
             'Nonsense_Mutation', 'De_novo_Start_OutOfFrame', 'Frame_Shift_Del', 'In_Frame_Del', 'Frame_Shift_Ins',
             'De_novo_Start_InFrame', 'Start_Codon_SNP', 'In_Frame_Ins', 'Nonstop_Mutation', 'Start_Codon_Del',
             'Stop_Codon_Del', 'Stop_Codon_Ins', 'Start_Codon_Ins']
    snps = ['SNP', 'DNP', 'TNP', 'DEL', 'INS', 'ONP']
    dfs = []
    cols = []
    for i in types:
        df_tmp = df.loc[df['type'] == i, ['group']]
        df_tmp = df_tmp.groupby(['group']).size().reset_index(name=i)
        df_tmp.index = df_tmp['group']
        df_tmp = df_tmp.loc[::, [i]]
        dfs.append(df_tmp)
        cols.append(i)
    for i in snps:
        df_tmp = df.loc[df['snp'] == i, ['group']]
        df_tmp = df_tmp.groupby(['group']).size().reset_index(name=i)
        df_tmp.index = df_tmp['group']
        df_tmp = df_tmp.loc[::, [i]]
        dfs.append(df_tmp)
        cols.append(i)
    df = df.groupby(['group']).size().reset_index(name=str_col + '_all')
    df.index = df['group']
    df = df.loc[::, []]
    for i in range(len(cols)):
        type = str_col + '_' + cols[i]
        df[type] = 0
        ids = dfs[i].index
        df.loc[ids, type] = dfs[i].loc[::, cols[i]].values.astype(int)
    return df


def fea_mut_mean(df_ori):
    df = df_ori.copy()
    df = df.groupby(['group', 'id']).size().reset_index(name='sample_count')
    df = df.drop_duplicates(subset=['group', 'id'], keep='first')
    df_mean = df.groupby('group').agg(q=('sample_count', 'mean')).reset_index()
    df_var = df.groupby('group').agg(q=('sample_count', 'var')).reset_index()
    df_mean.index = df_mean['group']
    df_var.index = df_var['group']
    df_mean = df_mean.loc[::, ['q']]
    df_var = df_var.loc[::, ['q']]
    df_mean.columns = ['sample_count_mean']
    df_var.columns = ['sample_count_var']
    df_all = pd.concat([df_mean, df_var], axis=1)
    return df_all


def fea_mut_gc(df_ori):
    df = df_ori.copy()
    df_mean = df.groupby('group').agg(q=('gc', 'mean')).reset_index()
    df_var = df.groupby('group').agg(q=('gc', 'var')).reset_index()
    df_mean.index = df_mean['group']
    df_var.index = df_var['group']
    df_mean = df_mean.loc[::, ['q']]
    df_var = df_var.loc[::, ['q']]
    df_mean.columns = ['gc_mean']
    df_var.columns = ['gc_var']
    df_all = pd.concat([df_mean, df_var], axis=1)
    return df_all


def fea_mut(df):
    tmp_dir = f'{root_path}/data/extract_fea/non-coding/tmp'
    fea_bed = tmp_dir + '/fea.bed'
    tmp_bed = tmp_dir + '/out.bed'
    sample_bed = 'chr_id_new.bed'
    df.to_csv(fea_bed, header=False, index=False, sep='\t')
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, fea_bed, tmp_bed)
    check_output(cmd, shell=True)
    df = pd.read_csv(tmp_bed, header=None, sep='\t', usecols=[3, 7, 8, 9, 10])
    df.columns = ['group', 'type', 'snp', 'id', 'gc']
    df_freq = df.copy().loc[::, ['group', 'id']]
    df_gc = df.copy().loc[::, ['group', 'gc']]
    df_gc_out = fea_mut_gc(df_gc)
    # df = df.drop_duplicates(subset=['group', 'id'], keep='first')
    df = df.loc[::, ['group', 'type', 'snp']]
    df = fea_mut_all(df, 'freq')
    df_add = fea_mut_mean(df_freq)
    df = pd.concat([df, df_add, df_gc_out], axis=1)
    return df


def fea_cna(df, col_name):
    tmp_dir = f'{root_path}/data/extract_fea/non-coding/tmp'
    fea_bed = tmp_dir + '/fea.bed'
    freq_bed = tmp_dir + '/sample.bed'
    tmp_bed = tmp_dir + '/out.bed'
    sample_bed = 'chr_id_new.bed'
    df.loc[::, ['chr', 'start', 'end', 'cna']].to_csv(fea_bed, header=False, index=False, sep='\t')
    df.loc[::, ['chr', 'start', 'end', 'id']].to_csv(freq_bed, header=False, index=False, sep='\t')
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, fea_bed, tmp_bed)
    print(cmd)
    check_output(cmd, shell=True)
    df = pd.read_csv(tmp_bed, header=None, sep='\t', usecols=[3, 7])
    df.columns = ['group', 'cna']
    df_mean = df.groupby('group').agg(q=('cna', 'mean')).reset_index()
    df_var = df.groupby('group').agg(q=('cna', 'var')).reset_index()
    df_mean.index = df_mean['group']
    df_var.index = df_var['group']
    df_mean = df_mean.loc[::, ['q']]
    df_var = df_var.loc[::, ['q']]
    df_mean.columns = [col_name + '_mean']
    df_var.columns = [col_name + '_var']
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, freq_bed, tmp_bed)
    print(cmd)
    check_output(cmd, shell=True)
    df_freq = pd.read_csv(tmp_bed, header=None, sep='\t', usecols=[3, 7])
    df_freq.columns = ['group', 'id']
    df_freq = df_freq.drop_duplicates(subset=['group', 'id'], keep='first')
    df_freq = df_freq.groupby(['group']).size().reset_index(name=col_name + '_freq')
    df_freq.index = df_freq['group']
    df_freq = df_freq.loc[::, [col_name + '_freq']]
    df = pd.concat([df_mean, df_var, df_freq], axis=1)
    print(df_mean.shape, df_var.shape, df_freq.shape, df.shape)
    cmd = "rm -f %s/*.bed" % tmp_dir
    check_output(cmd, shell=True)
    return df


def fea_rna(df):
    print(df.shape)
    df = np.log2(df + 1)
    means = df.mean(axis=1)
    vars = df.var(axis=1)
    df['exp_mean'] = means
    df['exp_var'] = vars
    df = df.loc[::, ['exp_mean', 'exp_var']]
    return df


def fea_CCLE():
    tmp_dir = f'{root_path}/data/extract_fea/non-coding/tmp'
    char_file = 'genome-characteristic.txt'
    df = pd.read_csv(char_file, header=0, sep='\t')
    df.fillna(df.mean(), inplace=True)
    df['expression_CCLE'] = df['expression_CCLE'].apply(lambda x: math.log2(x + 1))
    tmp_input = '%s/input.bed' % tmp_dir
    tmp_out = '%s/out.bed' % tmp_dir
    df.to_csv(tmp_input, index=False, header=False, sep='\t')
    sample_bed = 'chr_id_new.bed'
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, tmp_input, tmp_out)
    print(cmd)
    check_output(cmd, shell=True)
    df = pd.read_csv(tmp_out, header=None, sep='\t', usecols=[3, 7, 8])
    df.columns = ['id', 'rep_time', 'exp_CCLE']
    df = df.drop_duplicates(subset=['id'], keep='first')
    df.index = df['id']
    df.drop(['id'], axis=1, inplace=True)
    return df


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument("-i", dest='input', default="./chr_id.txt", help="clinvar_pos")
    parser.add_argument("-m", dest='mode', default="rna", help="clinvar_pos")#[cna,mut,rna]
    parser.add_argument("-t", dest='tumor', default="Pancan", help="clinvar_pos")  
    args = parser.parse_args()
    cancer_type=args.tumor
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    fea_data = ['mut', 'cna', 'rna']
    tumors_file = 'tumors.txt'
    samples_file = 'samples_info.csv'
    tumors_set = {'Pancan': 'Pancan'}
    plist = []
    for line in open(tumors_file, 'rt'):
        txt = line.rstrip().split('\t')
        tumors_set[txt[0]] = txt[1]
    path = f'{root_path}/results/non-coding/%s/' % tumors_set[args.tumor]
    if not exists(path):
        os.makedirs(path)
    df_samples = pd.read_csv(samples_file, header=0, sep=',',
                             usecols=['tumour_specimen_aliquot_id', 'normal_specimen_aliquot_id',
                                      'histology_abbreviation', 'submitted_donor_id'])
                                      
                                      
    if args.tumor != 'Pancan':
        df_samples = df_samples[df_samples['histology_abbreviation'] == args.tumor]
    
    else:
        pass

    sample_ids = df_samples.loc[::, 'tumour_specimen_aliquot_id'].values.astype(str)
    rna_ids = df_samples.loc[::, 'submitted_donor_id'].values.astype(str)
    samples = {}
    rnas = {}
    for id in sample_ids:
        samples[id] = 1     # samples
    for i in range(len(sample_ids)):
        rnas[rna_ids[i]] = sample_ids[i]
    dfs = []
    df0 = pd.read_csv(args.input, header=0, index_col=3, sep='\t')
    #print(df0.head(10))
    df0 = df0.loc[::, []]
    #print(df0.shape)
    df1 = pd.read_csv(args.input, header=0, sep='\t')
    id_tmp = df1['id'].values.tolist()
    ids = {}
    for id in id_tmp:
        ids[id] = 1
    outfile = './%s/%s.fea' % (path, args.mode)
    if args.mode == 'mut':
        input_file = 'final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
        df = pd.read_csv(input_file, header=0, sep='\t',
                         usecols=['Chromosome', 'Start_position', 'End_position', 'Variant_Classification',
                                  'Variant_Type', 'Tumor_Sample_Barcode', 'gc_content'])
        df.columns = ['chr', 'start', 'end', 'type', 'snp', 'id', 'gc']
        #print(samples.items())
        if args.tumor != 'Pancan':
            df = df[df['id'].isin(samples)]
        else:
            pass
        #print('wdwdwdwd',df.shape)
        X = fea_mut(df)
        # X = pd.concat([X1, X2], axis=1)
        print("extract mutation features...")
        print('dimension of mutation features:',X.shape)
        print("extract mutation features successfully")


    elif args.mode == 'cna':
        input_file = 'focal_input.rmcnv.pt_170207.seg.txt.gz'
        df = pd.read_csv(input_file, header=0, sep='\t',
                         usecols=['Chromosome', 'Start.bp', 'End.bp', 'Sample', 'Seg.CN'])
        df.columns = ['id', 'chr', 'start', 'end', 'cna']
        df = df.loc[::, ['chr', 'start', 'end', 'id', 'cna']]
        if args.tumor != 'Pancan':
            df = df.loc[df['id'].isin(samples), :]
        else:
            pass
        #print('wdwdwdwd',df.shape)
        #print(df.head())
        df_amp = df.loc[df['cna'] >= 1e-8, ::]
        df_del = df.loc[df['cna'] < -1e-8, ::]
        df_all = df.copy().loc[df['cna'] != 0, ::]
        X1 = fea_cna(df_amp, 'amp')
        X2 = fea_cna(df_del, 'del')
        df_all['cna'] = df_all['cna'].abs()
        X3 = fea_cna(df_all, 'abs')
        X = pd.concat([X1, X2, X3], axis=1)
        print("extract copy number features...")
        print('dimension of copy number features:',X.shape)
        print("extract copy number features successfully")
        
    elif args.mode == 'rna':
        list_bed = pd.read_csv('chr_id_new.bed', header=0, sep='\t').iloc[::, 3].values.tolist()
        input_file = 'tophat_star_fpkm_uq.v2_aliquot_gl_ncg.tsv.gz'
        df = pd.read_csv(input_file, header=0, index_col=0, sep='\t')
        rna_sample_file = 'rna-seq_summary.txt' 
        tcgas = {}
        df_samples = pd.read_csv(rna_sample_file, header=0, sep='\t', usecols=['aliquot_id', 'submitter_donor_id'])
        for idx, row in df_samples.iterrows():
            tcgas[row['aliquot_id']] = row['submitter_donor_id']
        l = list(df)
        #print(len(l))
        rna_samples = []
        other_samples = []
        for id in l:
            if id in tcgas and tcgas[id] in rnas:   
                rna_samples.append(id)              
            else: 
                other_samples.append(id)
        #print('-------')
        #print(len(rna_samples))
        #print('-------')
        rna_feas = list(df.index)
        fea_ids = {}
        for id in rna_feas:
            tmp = re.split('::', id)[2]
            tmp = tmp.split("_")[1]
            if tmp != 'nan':
                fea_ids[tmp] = id
        ids_sub = {}
        rna_fea_list = []
        ori_fea_list = []
        # rna feature ids to ori feature ids
        for id in list_bed:
            tmp = re.split('::', id)[3]
            if tmp in fea_ids:
                ids_sub[id] = fea_ids[tmp]
                rna_fea_list.append(fea_ids[tmp])
                ori_fea_list.append(id)
        #print(df.shape)
        df = df.loc[rna_fea_list, rna_samples] 
        #print(df.shape)
        cols_names = []
        for id in list(df):
            cols_names.append(rnas[tcgas[id]])
        df.index = ori_fea_list
        df.columns = cols_names
        X1 = fea_rna(df)
        X2 = fea_CCLE()
        X = pd.concat([X1, X2], axis=1)
        print("extract gene expression features...")
        print('dimension of gene expression features:',X.shape)
        print("extract gene expression features successfully")
        #print(X.head(10))
    x_ids = list(X.index)
    fea_ids = []
    for id in x_ids:
        if id in ids:
            fea_ids.append(id)
    cols = list(X)
    #print(cols)
    for col in cols:
        df0[col] = 0.0
        df0.loc[fea_ids, col] = X.loc[fea_ids, col].values.astype(float)
    df0 = df0.fillna(0.0)
    df0.index.name = 'geneid'
    #print(df0.head(10))
    #print(df0.shape)
    # df_tmp = df0.sort_values(by=['q'], ascending=[False])
    # print(df_tmp.head(5))
    df0.to_csv(outfile, header=True, index=True, sep='\t')


if __name__ == "__main__":
    main()
