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


def fea_seq_mean_var(df_ori, str):
    str_mean = str + '_mean'
    str_var = str + '_var'
    df = df_ori.copy()
    df_mean = df.groupby('group').agg(q=('score', 'mean')).reset_index()
    df_var = df.groupby('group').agg(q=('score', 'var')).reset_index()
    df_mean.index = df_mean['group']
    df_var.index = df_var['group']
    df_mean = df_mean.loc[::, ['q']]
    df_var = df_var.loc[::, ['q']]
    df_mean.columns = [str_mean]
    df_var.columns = [str_var]
    df_all = pd.concat([df_mean, df_var], axis=1)
    return df_all


def fea_mut(df):
    tmp_dir = './tmp/'
    fea_bed = tmp_dir + '/fea.bed'
    tmp_bed = tmp_dir + '/out.bed'
    # sample_bed = './chr_id.bed'
    sample_bed = './chr_id_new.bed'
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


def fea_mut2(df):
    tmp_dir = './tmp/'
    fea_bed = tmp_dir + '/fea.bed'
    tmp_bed = tmp_dir + '/out.bed'
    # sample_bed = './chr_id.bed'
    sample_bed = './chr_id_new.bed'
    df.to_csv(fea_bed, header=False, index=False, sep='\t')
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, fea_bed, tmp_bed)
    check_output(cmd, shell=True)
    df = pd.read_csv(tmp_bed, header=None, sep='\t', usecols=[3, 7, 8, 9, 10])
    df.columns = ['group', 'id', 'ref_3mer', 'ref_2mer_1', 'ref_2mer_2']
    df_3mer = df.copy().loc[::, ['group', 'ref_3mer']]
    df_3mer['fea'] = 1
    df_3mer = df_3mer.groupby(['group', 'ref_3mer']).sum().reset_index()
    fea_ref = df_3mer.pivot(index='group', columns='ref_3mer', values='fea')
    fea_ref = fea_ref.add_suffix('_ref')
    del df_3mer
    df_3mer = df.copy().loc[::, ['group', 'ref_2mer_1']]
    df_3mer['fea'] = 1
    df_3mer = df_3mer.groupby(['group', 'ref_2mer_1']).sum().reset_index()
    fea_21 = df_3mer.pivot(index='group', columns='ref_2mer_1', values='fea')
    fea_21 = fea_21.add_suffix('_ref21')
    del df_3mer
    df_3mer = df.copy().loc[::, ['group', 'ref_2mer_2']]
    df_3mer['fea'] = 1
    df_3mer = df_3mer.groupby(['group', 'ref_2mer_2']).sum().reset_index()
    fea_22 = df_3mer.pivot(index='group', columns='ref_2mer_2', values='fea')
    fea_22 = fea_22.add_suffix('_ref22')
    fea = pd.concat([fea_ref, fea_21, fea_22], axis=1).fillna(0)
    return fea


def fea_seq(df,method):
    df0 = df
    df0 = df0[~df0['chr'].isin(['X', 'Y'])]
    df0['chr'] = df0['chr'].astype(int)
    print('df0>>>',df0.head(10))
    print('df0>>>',df0.shape)
    method = method
    tmp_dir = './tmp/'
    fea_bed = tmp_dir + '/fea.bed'
    fea_tabix = tmp_dir + '/fea.tsv'
    fea_score = tmp_dir + '/fea.' + method
    tmp_bed = tmp_dir + '/out.bed'
    # sample_bed = './chr_id.bed'
    sample_bed = './chr_id_new.bed'
    # df.to_csv(fea_tabix, header=False, index=False, sep='\t')
    dvar_file = './../data/DVAR/hg19/hg19_DVAR.score.gz'
    cmd = "tabix %s -R %s >%s" % (dvar_file, fea_tabix, fea_score)
    # check_output(cmd, shell=True)
    df1 = pd.read_csv(fea_score, header=None, sep='\t')
    if method == 'cadd':
       df1.columns = ['chr', 'pos', 'ref', 'alt', 'score1', 'score2'] #cadd
    else: 
       df1.columns = ['chr', 'pos', 'ref', 'alt', 'score'] #css
    print('df1>>>',df1.head(10))
    print('df1>>>',df1.shape)
    print('df1type>>>', type(df1['chr'].values[0]))
    print('df1type>>>', type(df1['pos'].values[0]))
    print('df0type>>>', type(df0['chr'].values[1]))
    print('df0type>>>', type(df0['pos'].values[0]))
    #df0 = df0[~df0['chr'].isin(['X', 'Y'])]
    #df1 = df1[~df1['chr'].isin(['X', 'Y'])]
    #df0['chr'] = df0['chr'].astype(int)
    #df1['chr'] = df1['chr'].astype(int)
    # df0['pos'] = df0['pos'].astype(int)
    # df1['pos'] = df1['pos'].astype(int)
    df = pd.merge(df0, df1, on=['chr', 'pos'], how='inner')
    print('df>>>',df.head(10))
    print('df>>>',df.shape)
    df.insert(1, 'start', df['pos'])
    if method == 'cadd':
       df = df.loc[::, ['chr', 'start', 'pos', 'score2']] #cadd
    else: 
       df = df.loc[::, ['chr', 'start', 'pos', 'score']]  #css
    #df = df.loc[::, ['chr', 'start', 'pos', 'score2']]  #cadd  
    #df = df.loc[::, ['chr', 'start', 'pos', 'score']] #css
    print('df_start>>>',df.shape)
    df.to_csv(fea_bed, header=False, index=False, sep='\t')
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, fea_bed, tmp_bed)
    check_output(cmd, shell=True)
    df = pd.read_csv(tmp_bed, header=None, sep='\t', usecols=[3, 7])
    df.columns = ['group', 'score']
    fea = fea_seq_mean_var(df, method.upper())
    # fea = df.groupby('group').sum().reset_index()
    # fea.index = fea['group']
    # fea.drop(['group'], axis=1, inplace=True)
    fea = fea.fillna(0)
    return fea


def fea_cna(df, col_name):
    tmp_dir = './tmp/'
    fea_bed = tmp_dir + '/fea.bed'
    freq_bed = tmp_dir + '/sample.bed'
    tmp_bed = tmp_dir + '/out.bed'
    # sample_bed = './chr_id.bed'
    sample_bed = './chr_id_new.bed'
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
    tmp_dir = './tmp'
    char_file = '../data/genome-characteristic.txt'
    df = pd.read_csv(char_file, header=0, sep='\t')
    df.fillna(df.mean(), inplace=True)
    df['expression_CCLE'] = df['expression_CCLE'].apply(lambda x: math.log2(x + 1))
    tmp_input = '%s/input.bed' % tmp_dir
    tmp_out = '%s/out.bed' % tmp_dir
    df.to_csv(tmp_input, index=False, header=False, sep='\t')
    # sample_bed = './chr_id.bed'
    sample_bed = './chr_id_new.bed'
    cmd = "bedtools intersect -wb -a %s -b %s >%s" % (sample_bed, tmp_input, tmp_out)
    print(cmd)
    check_output(cmd, shell=True)
    df = pd.read_csv(tmp_out, header=None, sep='\t', usecols=[3, 7, 8])
    df.columns = ['id', 'rep_time', 'exp_CCLE']
    df = df.drop_duplicates(subset=['id'], keep='first')
    df.index = df['id']
    df.drop(['id'], axis=1, inplace=True)
    return df


def compute_k_mer_const(k_mer_len):
    nts = ['A', 'C', 'G', 'T']
    kmers = []
    kmers.append('')
    l = 0
    while l < k_mer_len:
        imers = []
        for imer in kmers:
            for nt in nts:
                imers.append(imer + nt)
        kmers = imers
        l += 1
    max_features = len(kmers)
    rc = {'A': 'T', 'G': 'C', 'C': 'G', 'T': 'A'}
    revcomp = lambda s: ''.join([rc[s[i]] for i in range(len(s) - 1, -1, -1)])
    kmer_id_dict = {}
    for i in range(len(kmers)):
        kmer_id_dict[kmers[i]] = i
    nb_word = 0
    mapping_dict = {}
    for k_mer_id in kmers:
        rc_id = revcomp(k_mer_id)
        if rc_id in mapping_dict:
            mapping_dict[k_mer_id] = rc_id
        else:
            mapping_dict[k_mer_id] = k_mer_id
            nb_word += 1
    return nb_word, kmers, mapping_dict


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='input', default="./chr_id.txt", help="clinvar_pos")
    parser.add_argument("-m", dest='mode', default="rna", help="clinvar_pos")#[cadd,cna,css,mut,mut2,rna]
    parser.add_argument("-t", dest='tumor', default="Stomach-AdenoCA", help="clinvar_pos")  
    args = parser.parse_args()
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    fea_data = ['mut', 'cna', 'rna']
    tumors_file = './tumors.txt'
    samples_file = '../data/ICGC/samples_info.csv'
    tumors_set = {'Pancan': 'Pancan'}
    plist = []
    for line in open(tumors_file, 'rt'):
        txt = line.rstrip().split('\t')
        tumors_set[txt[0]] = txt[1]
    path = './%s/' % tumors_set[args.tumor]
    if not exists(path):
        os.makedirs(path)
    df_samples = pd.read_csv(samples_file, header=0, sep=',',
                             usecols=['tumour_specimen_aliquot_id', 'normal_specimen_aliquot_id',
                                      'histology_abbreviation', 'submitted_donor_id'])
                                      
                                      
    if args.tumor != 'Pancan':
        df_samples = df_samples[df_samples['histology_abbreviation'] == args.tumor]
        print('df_samples11111111111111111111',df_samples.shape)
    
    else:
        pass
        # plist = ['Ovary-AdenoCA', 'CNS-Medullo', 'Biliary-AdenoCA', 'Breast-AdenoCA', 'CNS-PiloAstro', 'Bone-Osteosarc',
        #          'Myeloid-AML', 'Bone-Epith', 'Liver-HCC', 'Prost-AdenoCA', 'Panc-Endocrine', 'Breast-LobularCA',
        #          'Myeloid-MDS', 'Head-SCC', 'Eso-AdenoCA', 'Stomach-AdenoCA', 'Panc-AdenoCA', 'Bone-Cart',
        #          'Breast-DCIS', 'Myeloid-MPN', 'Kidney-RCC']
        # df_samples = df_samples[df_samples['histology_abbreviation'].isin(plist)]
    sample_ids = df_samples.loc[::, 'tumour_specimen_aliquot_id'].values.astype(str)
    rna_ids = df_samples.loc[::, 'submitted_donor_id'].values.astype(str)
    samples = {}
    rnas = {}
    for id in sample_ids:
        samples[id] = 1     # samples
    for i in range(len(sample_ids)):
        rnas[rna_ids[i]] = sample_ids[i]
    dfs = []
    # df_tmp = pd.read_csv('./chr_id.txt', header=None, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
    # df_tmp.columns = ['chr', 'start', 'end']
    # df_tmp['chr'] = df_tmp['chr'].apply(lambda x: str(x).replace("chr", ""))
    # df_tmp['id'] = df_tmp.index
    # df_tmp = df_tmp.sort_values(by=['chr', 'start'], ascending=[True, True])
    # df_tmp.to_csv('chr_id.bed', header=False, index=False, sep='\t')
    # #df_tmp.to_csv('chr_id.txt', header=True, index=False, sep='\t')
    # return
    df0 = pd.read_csv(args.input, header=0, index_col=3, sep='\t')
    print(df0.head(10))
    df0 = df0.loc[::, []]
    print(df0.shape)
    df1 = pd.read_csv(args.input, header=0, sep='\t')
    id_tmp = df1['id'].values.tolist()
    ids = {}
    for id in id_tmp:
        ids[id] = 1
    outfile = './%s/%s.fea' % (path, args.mode)
    if args.mode == 'mut':
        input_file = '../data/ICGC/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
        df = pd.read_csv(input_file, header=0, sep='\t',
                         usecols=['Chromosome', 'Start_position', 'End_position', 'Variant_Classification',
                                  'Variant_Type', 'Tumor_Sample_Barcode', 'gc_content'])
        df.columns = ['chr', 'start', 'end', 'type', 'snp', 'id', 'gc']
        #print(samples.items())
        df = df[df['id'].isin(samples)]
        print('wdwdwdwd',df.shape)
        X = fea_mut(df)
        # X = pd.concat([X1, X2], axis=1)
        print(X.shape)
    elif args.mode == 'mut2':
        _, _, mer_dict_3 = compute_k_mer_const(3)
        _, _, mer_dict_2 = compute_k_mer_const(2)
        input_file = '../data/ICGC/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
        df = pd.read_csv(input_file, header=0, sep='\t', usecols=['Chromosome', 'Start_position', 'End_position',
                                                                  'Variant_Type', 'Tumor_Sample_Barcode',
                                                                  'Tumor_Seq_Allele2', 'ref_context'])
        df.columns = ['chr', 'start', 'end', 'snp', 'alt', 'id', 'ref']
        df = df.loc[df['snp'] == 'SNP', :]
        df = df.loc[df['id'].isin(samples), :]
        print('wdwdwdwd',df.shape)
        # df["alt_3mer"] = df.apply(lambda row: (row["ref"][9:10] + row['alt'] + row["ref"][11:12]).upper(), axis=1)
        # df['alt_3mer'] = df['alt_3mer'].map(mer_dict_3)
        print('mut2_df>>>', df.head(5))
        df["ref_3mer"] = df.apply(lambda row: (row["ref"][9:12]).upper(), axis=1)
        df['ref_3mer'] = df['ref_3mer'].map(mer_dict_3)
        df["ref_2mer_1"] = df.apply(lambda row: (row["ref"][9:11]).upper(), axis=1)
        df['ref_2mer_1'] = df['ref_2mer_1'].map(mer_dict_2)
        df["ref_2mer_2"] = df.apply(lambda row: (row["ref"][10:12]).upper(), axis=1)
        df['ref_2mer_2'] = df['ref_2mer_2'].map(mer_dict_2)
        df = df.loc[:, ['chr', 'start', 'end', 'id', 'ref_3mer', 'ref_2mer_1', 'ref_2mer_2']]
        X = fea_mut2(df)
        # X = pd.concat([X1, X2], axis=1)
        print(X.shape)
    elif args.mode == 'cadd':
        nts = ['A', 'C', 'G', 'T']
        input_file = '../data/ICGC/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
        df = pd.read_csv(input_file, header=0, sep='\t', usecols=['Chromosome', 'Start_position', 'End_position',
                                                                   'Variant_Type', 'Reference_Allele',
                                                                   'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode',
                                                                   ])
        df.columns = ['chr', 'start', 'pos', 'snp', 'ref', 'alt', 'id']
        df = df.loc[df['snp'] == 'SNP', :]
        df = df.loc[df['id'].isin(samples), :]
        print('wdwdwdwd',df.shape)
        # print(df.head())
        df = df[df['ref'].isin(nts)]
        df = df[df['alt'].isin(nts)]
        df = df.loc[:, ['chr', 'pos', 'ref', 'alt']]
        df = df.drop_duplicates(subset=['chr', 'pos'], keep='first')
        df = df.sort_values(by=['chr', 'pos'], ascending=[True, True])
        #df = None
        X = fea_seq(df,args.mode)
        # X = pd.concat([X1, X2], axis=1)
        print(X.shape)
        
    elif args.mode == 'css':
        nts = ['A', 'C', 'G', 'T']
        input_file = '../data/ICGC/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz'
        df = pd.read_csv(input_file, header=0, sep='\t', usecols=['Chromosome', 'Start_position', 'End_position',
                                                                   'Variant_Type', 'Reference_Allele',
                                                                   'Tumor_Seq_Allele2', 'Tumor_Sample_Barcode',
                                                                   ])
        df.columns = ['chr', 'start', 'pos', 'snp', 'ref', 'alt', 'id']
        df = df.loc[df['snp'] == 'SNP', :]
        df = df.loc[df['id'].isin(samples), :]
        print('wdwdwdwd',df.shape)
        # print(df.head())
        df = df[df['ref'].isin(nts)]
        df = df[df['alt'].isin(nts)]
        df = df.loc[:, ['chr', 'pos', 'ref', 'alt']]
        df = df.drop_duplicates(subset=['chr', 'pos'], keep='first')
        df = df.sort_values(by=['chr', 'pos'], ascending=[True, True])
        #df = None
        X = fea_seq(df,args.mode)
        # X = pd.concat([X1, X2], axis=1)
        print(X.shape)

    elif args.mode == 'cna':
        input_file = '../data/ICGC/focal_input.rmcnv.pt_170207.seg.txt.gz'
        df = pd.read_csv(input_file, header=0, sep='\t',
                         usecols=['Chromosome', 'Start.bp', 'End.bp', 'Sample', 'Seg.CN'])
        df.columns = ['id', 'chr', 'start', 'end', 'cna']
        df = df.loc[::, ['chr', 'start', 'end', 'id', 'cna']]
        df = df.loc[df['id'].isin(samples), :]
        print('wdwdwdwd',df.shape)
        print(df.head())
        df_amp = df.loc[df['cna'] >= 1e-8, ::]
        df_del = df.loc[df['cna'] < -1e-8, ::]
        df_all = df.copy().loc[df['cna'] != 0, ::]
        X1 = fea_cna(df_amp, 'amp')
        X2 = fea_cna(df_del, 'del')
        df_all['cna'] = df_all['cna'].abs()
        X3 = fea_cna(df_all, 'abs')
        X = pd.concat([X1, X2, X3], axis=1)
        print(X.head(5))
    elif args.mode == 'rna':
        list_bed = pd.read_csv('chr_id_new.bed', header=0, sep='\t').iloc[::, 3].values.tolist()
        input_file = '../data/ICGC/tophat_star_fpkm_uq.v2_aliquot_gl_ncg.tsv.gz'
        df = pd.read_csv(input_file, header=0, index_col=0, sep='\t')
        rna_sample_file = './rna-seq_summary.txt' 
        tcgas = {}
        df_samples = pd.read_csv(rna_sample_file, header=0, sep='\t', usecols=['aliquot_id', 'submitter_donor_id'])
        for idx, row in df_samples.iterrows():
            tcgas[row['aliquot_id']] = row['submitter_donor_id']
        l = list(df)
        print(len(l))
        rna_samples = []
        other_samples = []
        for id in l:
            if id in tcgas and tcgas[id] in rnas:   
                rna_samples.append(id)              
            else: 
                other_samples.append(id)
        print('-------')
        print(len(rna_samples))
        print('-------')
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
        print(df.shape)
        df = df.loc[rna_fea_list, rna_samples] 
        print(df.shape)
        cols_names = []
        for id in list(df):
            cols_names.append(rnas[tcgas[id]])
        df.index = ori_fea_list
        df.columns = cols_names
        X1 = fea_rna(df)
        X2 = fea_CCLE()
        X = pd.concat([X1, X2], axis=1)
        print(X.head(10))
    x_ids = list(X.index)
    fea_ids = []
    for id in x_ids:
        if id in ids:
            fea_ids.append(id)
    cols = list(X)
    print(cols)
    for col in cols:
        df0[col] = 0.0
        df0.loc[fea_ids, col] = X.loc[fea_ids, col].values.astype(float)
    df0 = df0.fillna(0.0)
    df0.index.name = 'geneid'
    print(df0.head(10))
    print(df0.shape)
    # df_tmp = df0.sort_values(by=['q'], ascending=[False])
    # print(df_tmp.head(5))
    df0.to_csv(outfile, header=True, index=True, sep='\t')


if __name__ == "__main__":
    main()
