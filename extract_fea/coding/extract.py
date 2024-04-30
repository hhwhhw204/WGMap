import pandas as pd
import numpy as np
import argparse


def mutation(mut_file):
    all_data = "clinical_PANCAN_patient_with_followup.tsv"
    all_sample = pd.read_csv(all_data, header=0, sep='\t', encoding='gbk',low_memory=False)
    all_sample_ids = set(all_sample['bcr_patient_barcode'].tolist())
    result_ids = [x[:12] for x in all_sample_ids]

    df = pd.read_csv(mut_file, sep=',')
    df = df[df['Tumor_Sample_Barcode'].apply(lambda x: x[:12]).isin(result_ids)]
    # Variant_Classification
    vc_df = df.groupby(['Hugo_Symbol', 'Variant_Classification']).size().reset_index(name='num')
    vc_df = vc_df.set_index(['Hugo_Symbol', 'Variant_Classification'])['num']
    vc_df = vc_df.unstack()
    vc_df = vc_df.rename_axis(columns=None)
    vc_df = vc_df.reset_index()
    vc_df = vc_df.fillna(0)
    # Variant_Type
    vt_df = df.groupby(['Hugo_Symbol', 'Variant_Type']).size().reset_index(name='num')
    vt_df = vt_df.set_index(['Hugo_Symbol', 'Variant_Type'])['num']
    vt_df = vt_df.unstack()
    vt_df = vt_df.rename_axis(columns=None)
    vt_df = vt_df.reset_index()
    vt_df = vt_df.fillna(0)
    # concat
    vt_df = vt_df.drop(columns=['Hugo_Symbol'])
    vc_vt_df = pd.concat([vc_df, vt_df], axis=1)
    # calculate pnum
    df = df.drop_duplicates(subset=['Hugo_Symbol', 'Tumor_Sample_Barcode'])
    df = df.groupby(['Hugo_Symbol', 'Tumor_Sample_Barcode']).size().reset_index(name='num')
    df = df.groupby('Hugo_Symbol').size().reset_index(name='pnum').drop(columns=['Hugo_Symbol'])
    df.columns = df.columns.str.strip()
    # concat
    merge_df = pd.concat([vc_vt_df, df], axis=1)
    merge_df.to_csv('/root/capsule/results/coding_fea/mut_fea.csv', index=False, sep=',')


def copy_num(mut_file,cnv_file):
    all_data = "clinical_PANCAN_patient_with_followup.tsv"
    all_sample = pd.read_csv(all_data, header=0, sep='\t', encoding='gbk',low_memory=False)
    all_sample_ids = set(all_sample['bcr_patient_barcode'].tolist())
    result_ids = [x[:12] for x in all_sample_ids]

    mut = pd.read_csv(mut_file, sep=',',header=0, low_memory=False)
    mut.loc[mut[(mut.Chromosome == 'X')].index.tolist(), 'Chromosome'] = '23'
    mut.loc[mut[(mut.Chromosome == 'Y')].index.tolist(), 'Chromosome'] = '23'
    mut.drop(mut.loc[mut['Chromosome'] == 'M'].index, inplace=True)
    mut = mut[['Hugo_Symbol', 'Chromosome', 'Start_Position', 'End_Position', 'Tumor_Sample_Barcode']]
    mut[['Chromosome']] = mut[['Chromosome']].astype(np.str_)
    mut[['Start_Position']] = mut[['Start_Position']].astype(np.int32)
    mut[['End_Position']] = mut[['End_Position']].astype(np.int32)
    mut[['Tumor_Sample_Barcode']] = mut[['Tumor_Sample_Barcode']].astype(np.str_)
    mut['New_Sample'] = mut['Tumor_Sample_Barcode'].apply(lambda x: x[:12]).tolist()

    copnum = pd.read_csv(cnv_file, header=0, sep='\t', low_memory=False)
    copnum[['Chromosome']] = copnum[['Chromosome']].astype(np.str_)
    copnum[['Start']] = copnum[['Start']].astype(np.int32)
    copnum[['End']] = copnum[['End']].astype(np.int32)
    copnum['New_Sample'] = copnum['Sample'].apply(lambda x: x[:12]).tolist()

    merge_data = pd.merge(mut, copnum, on=['New_Sample', 'Chromosome'])
    merge_data = merge_data[merge_data["Start_Position"] > merge_data["Start"]]
    merge_data = merge_data[merge_data["End_Position"] < merge_data["End"]]
    merge_data = merge_data[merge_data['New_Sample'].isin(result_ids)]

    df = merge_data[['Hugo_Symbol', 'Segment_Mean']]
    new_up = df.loc[df['Segment_Mean'] >= 0]
    new_down = df.loc[df['Segment_Mean'] < 0]
    df_mean = df.groupby('Hugo_Symbol').mean()
    df_var = df.groupby('Hugo_Symbol').var()
    df = pd.concat([df_mean, df_var], axis=1)
    df.columns = ['all_mean', 'all_var']

    df_up_mean = new_up.groupby('Hugo_Symbol').mean()
    df_up_var = new_up.groupby('Hugo_Symbol').var()
    df_up = pd.concat([df_up_mean, df_up_var], axis=1)
    df_up.columns = ['up_mean', 'up_var']

    df_down_mean = new_down.groupby('Hugo_Symbol').mean()
    df_down_var = new_down.groupby('Hugo_Symbol').var()
    df_down = pd.concat([df_down_mean, df_down_var], axis=1)
    df_down.columns = ['down_mean', 'down_var']
    new_data = pd.concat([df, df_up, df_down], axis=1)

    for column in list(new_data.columns[new_data.isnull().sum() > 0]):
        mean_val = new_data[column].mean()
        new_data[column].fillna(mean_val, inplace=True)
    new_data = new_data.reset_index()
    new_data.to_csv("/root/capsule/results/coding_fea/cnv_fea.csv",index=False)
    return new_data


def change_df(df):
    arr = df.values
    new_df = pd.DataFrame(arr[1:, 1:], index=arr[1:, 0], columns=arr[0, 1:])
    new_df.index.name = arr[0, 0]
    return new_df

def expression(exp_file):
    all_data = "clinical_PANCAN_patient_with_followup.tsv"
    all_sample = pd.read_csv(all_data, header=0, sep='\t', encoding='gbk',low_memory=False)
    all_sample_ids = set(all_sample['bcr_patient_barcode'].tolist())

    chunk = pd.read_csv(exp_file, sep="\t",low_memory=False)
    chunk['gene_id'] = [i[0] for i in chunk['gene_id'].str.split('|')]
    chunk = chunk[chunk['gene_id'] != '?']
    chunk = chunk.T.reset_index()
    chunk = change_df(chunk).reset_index()
    chunk['gene_id'] = chunk['gene_id'].apply(lambda x: x[:12]).tolist()
    chunk = chunk[chunk['gene_id'].isin(all_sample_ids)]
    chunk = chunk.T.reset_index()
    chunk = change_df(chunk)
    row_mean = chunk.mean(axis=1, skipna=True).astype('float')
    row_var = chunk.var(axis=1, skipna=True).astype('float')
    row_mean = np.log(row_mean + 1)
    row_var = np.log(row_var + 1)
    df = pd.DataFrame(index=row_mean.index)
    df['exp_mean'] = row_mean
    df['exp_var'] = row_var
    df.to_csv("/root/capsule/results/coding_fea/exp_fea.csv")


def main():
    parser = argparse.ArgumentParser(description='extract coding pancan fea')
    parser.add_argument("-t", dest='type', default="exp", choices=["mut","cnv","exp"],help="feature type")
    args = parser.parse_args()

    if args.type == 'mut':
        print("extract mutation fea..")
        mut_file = 'mut.csv'
        mutation(mut_file)

    if args.type == 'cnv':
        print("extract copy number fea..")
        mut_file = 'mut.csv'
        cnv_file = 'cnv_file'
        copy_num(mut_file,cnv_file)

    if args.type == 'exp':
        print("extract gene expression fea..")
        exp_file = "exp_file"
        expression(exp_file)


if __name__ == "__main__":
    main()
