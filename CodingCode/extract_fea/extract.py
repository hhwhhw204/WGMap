import argparse
import sys
import numpy as np
import pandas as pd
import math
import pysam
from optparse import OptionParser
import os


def trans_log(a):
    return math.log(a + 1)

def generate_exp():
    parser = argparse.ArgumentParser(description='Build Data Set.')
    parser.add_argument("-i", dest='input',
                        default="EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.tsv",
                        help="input file")
    parser.add_argument("-p", dest='out_path', default="rna.csv", help="clinvar")
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep='\t', index_col=0, header=0)
    #df = df.applymap(trans_log)
    df_var = df.std(axis=1, ddof=0)
    df = df.mean(axis=1)
    genes = list(df.index)
    mean_set ={}
    var_set ={}
    for key in genes:
        gene_key = key.rstrip().split("|")[0]
        if gene_key != '?':
            if gene_key in mean_set:
                mean_set[gene_key].append(df.at[key])
            else:
                mean_set[gene_key] = [df.at[key]]
            if gene_key in var_set:
                var_set[gene_key].append(df_var.at[key])
            else:
                var_set[gene_key] = [df_var.at[key]]
    fp = open(args.out_path, 'wt')
    fp.write("%s\t%s\t%s\n" % ("gene", "rna_mean", "rna_std"))
    for key in mean_set:
        fp.write("%s\t%f\t%f\n" % (key, np.mean(mean_set[key]),np.mean(var_set[key])))
    fp.close()
    # cmd0 = "bgzip -c %s > %s.gz" %(args.genome_out, args.genome_out)
    # check_output(cmd0, shell=True)
    # cmd1 = "tabix -s1 -b2 -e2 %s.gz" %(args.genome_out)
    # check_output(cmd1, shell=True)

def norm_0_1(a):
    if a != 0:
        return 1
    else:
        return 0


def generate_methylation():
    parser = argparse.ArgumentParser(description='Build Data Set.')
    parser.add_argument("-i", dest='input',
                        default="jhu-usc.edu_PANCAN_merged_HumanMethylation27_HumanMethylation450.betaValue_whitelisted.tsv",
                        help="input file")
    parser.add_argument("-r", dest='ref',
                        default="cpg2gene_final.txt",
                        help="ref file")
    parser.add_argument("-p", dest='out_path', default="methylation.csv", help="clinvar")
    args = parser.parse_args()
    cgs ={}
    methylation_mean_set ={}
    methylation_var_set = {}
    df_ref = pd.read_csv(args.ref, sep='\t', index_col=0, header=0)
    for index, row in df_ref.iterrows():
        cgs[index] = row['gene']
    df = pd.read_csv(args.input, sep='\t', index_col=0, header=0)
    df_mean = df.mean(axis=1)
    df_mean.columns = ['mean']
    df_std = df.std(axis=1, ddof=0)
    df_std.columns = ['std']
    for key in df_mean.index:   #合并数据 用均值填充空值
        if key in cgs:
            if cgs[key] in methylation_mean_set:
                methylation_mean_set[cgs[key]].append(df_mean.at[key])
            else:
                methylation_mean_set[cgs[key]] = [df_mean.at[key]]
            if cgs[key] in methylation_var_set:
                methylation_var_set[cgs[key]].append(df_std.at[key])
            else:
                methylation_var_set[cgs[key]] = [df_std.at[key]]
    fp = open(args.out_path, 'wt')
    fp.write("%s\t%s\t%s\n" % ("gene", "methylation_mean", "methylation_var"))
    for key in methylation_mean_set:
        fp.write("%s\t%f\t%f\n" % (key, np.mean(methylation_mean_set[key]), np.mean(methylation_var_set[key])))
    fp.close()



def amp(a):
    if a > 0:
        return a
    else:
        return 0


def down(a):
    if a < 0:
        return a
    else:
        return 0


def generate_cna():
    parser = argparse.ArgumentParser(description='Build Data Set.')
    parser.add_argument("-i", dest='input',
                        default="GISTIC.focal_data_by_genes.conf_95.txt.gz",
                        help="input file")
    parser.add_argument("-p", dest='out_path', default="./fea/scna.csv", help="clinvar")
    args = parser.parse_args()
    df = pd.read_csv(args.input, sep='\t', index_col=0, header=0)
    df.drop(['Locus ID', 'Cytoband'], axis=1, inplace=True)
    df_amp = df.applymap(amp)
    df_del = df.applymap(down)
    df_amp_mean = df_amp.mean(axis=1)
    df_amp_var = df_amp.std(axis=1, ddof=0)  #ddof=0 表示标准差为有偏
    df_del_mean = df_del.mean(axis=1)
    df_del_var = df_del.std(axis=1, ddof=0)
    genes = list(df.index)
    df_amp_mean_set = {}
    df_amp_var_set = {}
    df_del_mean_set = {}
    df_del_var_set = {}
    for key in genes:
        gene_key = key.rstrip().split("|")[0]
        if gene_key != '?':
            if gene_key in df_amp_mean_set:
                df_amp_mean_set[gene_key].append(df_amp_mean.at[key])
            else:
                df_amp_mean_set[gene_key] = [df_amp_mean.at[key]]
            if gene_key in df_amp_var_set:
                df_amp_var_set[gene_key].append(df_amp_var.at[key])
            else:
                df_amp_var_set[gene_key] = [df_amp_var.at[key]]
            if gene_key in df_del_mean_set:
                df_del_mean_set[gene_key].append(df_del_mean.at[key])
            else:
                df_del_mean_set[gene_key] = [df_del_mean.at[key]]
            if gene_key in df_del_var_set:
                df_del_var_set[gene_key].append(df_del_var.at[key])
            else:
                df_del_var_set[gene_key] = [df_del_var.at[key]]
    fp = open(args.out_path, 'wt')
    fp.write("%s\t%s\t%s\t%s\t%s\n" % ("gene", "scna_amp_mean", "scna_amp_std", "scna_del_mean", "scna_del_std"))
    for key in df_amp_mean_set:
        fp.write("%s\t%f\t%f\t%f\t%f\n" % (
        key, np.mean(df_amp_mean_set[key]), np.mean(df_amp_var_set[key]), np.mean(df_del_mean_set[key]),
        np.mean(df_del_var_set[key])))
    fp.close()
    # cmd0 = "bgzip -c %s > %s.gz" %(args.genome_out, args.genome_out)
    # check_output(cmd0, shell=True)
    # cmd1 = "tabix -s1 -b2 -e2 %s.gz" %(args.genome_out)
    # check_output(cmd1, shell=True)



def generate_cadd():
    parser = OptionParser()
    parser.add_option("-p", "--path", dest="cadd", default="./input/whole_genome_SNVs.tsv.gz")      
    parser.add_option("-i", "--input", dest="input", default="./input/mc3_sample.csv",              # mc3.maf处理得到csv
                    help="Read variants from vcf file (default stdin)")
    parser.add_option("-o", "--found_out", dest="found_out", help="Write found variants to file (default: stdout)",default="./out/cadd.score")
    (options, args) = parser.parse_args()

    stdin = pd.read_csv(options.input,sep=",")
    found_out = open(options.found_out, 'wt')

    fpos, fref, falt = 1, 2, 3
    if os.path.exists(options.cadd) and os.path.exists(options.cadd + ".tbi"):
        filename = options.cadd
        caddTabix = pysam.Tabixfile(filename, 'r')

    for i in range(0,len(stdin)):       # 3427680
        found_cadd = False
        fields = stdin.loc[i]
        chrom = str(fields[0])
        pos = int(fields[1])
        lref, allele = fields[4], fields[5]

        for regionHit in caddTabix.fetch(chrom, pos - 1, pos):
            vfields = regionHit.rstrip().split('\t')
            if (vfields[fref] == lref) and (vfields[falt] == allele) and (vfields[fpos] == fields[2]):
                # found_out.write(line + '\t' +vfields[-1] + "\n")
                found_cadd = True

    stdin.close()
    found_out.close()
