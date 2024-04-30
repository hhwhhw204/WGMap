# WGMap

WGMap is a cutting-edge machine learning framework that integrates multi-omics data (somatic mutations, copy number variations, gene expression, and DNA methylation) from large tumor cohorts (TCGA, PCAWG) to comprehensively identify genome-wide anti-cancer targets.A thorough genomic analysis within WGMap unveils a vast landscape of potential targets: 225 coding and a remarkable 1,752 non-coding genes with pan-cancer potential, alongside additional cancer-specific targets.

![wgmap](./WGMap.png)


# coding
echo "To run efficiently, we do not retrain the neural network. If you need to download the original data and retrain the neural network, change the -m predict parameter from the command line to train"

echo "Cross validation performance and predicted anti-cancer target genes of coding pan-cancer analysis on tier1 dataset"

``cd /code/CodingCode/pancan_tier1/code/``

``python dm_xgb_pancan_train.py -m predict``

``python dm_xgb_pancan_predict.py -m predict``


echo "Cross validation performance and predicted anti-cancer target genes of coding pan-cancer analysis on tier2 dataset"

``cd /code/CodingCode/pancan_tier2/code/``

``python semi_trans_pancan_train.py  -m predict``

``python semi_trans_pancan_predict.py  -m predict``  


echo "Cross validation performance and predicted anti-cancer target genes of coding specific-type-cancer analysis on tier1 datasets"

``cd /code/CodingCode/ecancer_tier1/code/``

``python dm_xgb_ecancer_train.py -m predict -t prad``

``python dm_xgb_ecancer_predict.py -m predict -t prad`` 


echo "Cross validation performance and predicted anti-cancer target genes of coding specific-type-cancer analysis on tier2 datasets"

``cd /code/CodingCode/ecancer_tier2/code/``

``python semi_trans_ecancer_train.py -m predict -t prad``

``python semi_trans_ecancer_predict.py -m predict -t prad``


# extract pancan fea
echo "We provide the coding omic features of mutation, CNV, and gene expression"

``cd /data/extract_fea/coding``

``mkdir /root/capsule/results/coding_fea/``

``python extract.py -t mut``

``python extract.py -t cnv``

``python extract.py -t exp``


# noncoding
echo "Cross validation performance and predicted anti-cancer target genes of non-coding pan-cancer analysis on tier1 dataset"

``cd /code/Non-codingCode/pancan_tier1/code``

``python WGDiffusion_train.py -m train``

``python WGDiffusion_predict.py -m pred``


echo "Cross validation performance and predicted anti-cancer target genes of non-coding pan-cancer analysis on tier2 dataset"

``cd /code/Non-codingCode/pancan_tier2/code``

``python WGFormer_train.py -m train``

``python WGFormer_predict.py -m pred``


echo "Cross validation performance and predicted anti-cancer target genes of non-coding specific-type-cancer analysis on tier1 datasets"

``cd /code/Non-codingCode/ecancer_tier1/code``

``python WGDiffusion_train.py -m train -t 'OV'``

``python WGDiffusion_predict.py -m pred -t 'OV'``


echo "Cross validation performance and predicted anti-cancer target genes of non-coding specific-type-cancer analysis on tier2 datasets"

``cd /code/Non-codingCode/ecancer_tier2/code``

``python WGFormer_train.py -m train -t 'STAD'``

``python WGFormer_predict.py -m pred -t 'LIHC'``


# extract fea (pan-cancer and specific-cancer)
echo "We provide the non-coding omic features of mutation, CNV, and gene expression"

``cd /data/extract_fea/non-coding``

``python extract_fea.py -m mut -t Pancan``

``python extract_fea.py -m cna -t Pancan``

``python extract_fea.py -m rna -t Pancan``

