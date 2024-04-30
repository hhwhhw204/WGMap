import argparse
import sys
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from math import sqrt
import numpy as np
import random
import pandas as pd
import re
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler  # 标准化工具
from sklearn import preprocessing
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score
import warnings
import torch
from sklearn.metrics import auc
import os
import math
from scipy.stats import fisher_exact
from scipy import stats
import joblib

warnings.filterwarnings('ignore')
seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
root_path ='../../../../'

#找roc的最优阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def fisher_ex(a, b, c, d):
    _, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    #p1 = -math.log10(pvalue)
    return pvalue

def build_set(cancer_type, all_list):
  pos_ids_noncds = []
  rna_ids = []
  neg_ids_noncds = []
  test_ids_ex = []
  tjumirna_list = []
  pos_path = f'{root_path}/code/Non-codingCode/ecancer_tier2/data/{cancer_type}.csv'
  df_pos_noncds = pd.read_csv(pos_path, sep=',')
  pos_ids_noncds = df_pos_noncds['geneid'].values.tolist()
  pos_ids = list(set(pos_ids_noncds))
  #随机选择负样本 前提是排除可能的正样本
  tumor_list = ['BRCA', 'ESCA', 'LIHC', 'OV',
          'STAD', 'PRAD', 'HNSC']
  excludes = []
  for i in tumor_list:
    if i != cancer_type:
      path = f'{root_path}/code/Non-codingCode/ecancer_tier2/data/{i}.csv'
      exclude = pd.read_csv(path, sep=',')
      exclude = exclude['geneid'].values.tolist()
      for j in exclude:
        excludes.append(j)
  #去重
  excludes = list(set(excludes))
  excludes_ids = pos_ids + excludes
  df_all_list = pd.DataFrame(all_list)
  df_all_list.columns = ['id']
  df_ffu = df_all_list[~df_all_list['id'].isin(excludes_ids)]
  all_list_ffu = df_ffu['id'].values.tolist()
  for id in all_list_ffu:
    tmps_ffu = re.split('::', id)
    reg_ffu = tmps_ffu[0]
    reg_ffu1 = tmps_ffu[1]
    if 'tjumirna' in reg_ffu1:
      tjumirna_list.append(id)
    if 'cds' not in reg_ffu and 'tjumirna' not in reg_ffu1:# and 'trnascanse' not in reg_ffu1 and 'snornabase' not in reg_ffu1 and 'mitranscriptome' not in reg_ffu1:
      neg_ids_noncds.append(id) #非编码区其他负样本基因
  df_neg_ids_noncds = pd.DataFrame(neg_ids_noncds)
  df_fu = df_neg_ids_noncds.sample(n=len(pos_ids)*10,random_state=8,replace = False) #8
  neg_ids = df_fu[0].values.tolist()

  for id in all_list:
    tmps = re.split('::', id)
    gene = tmps[2]
    reg = tmps[0]
    reg1 = tmps[1]
    if 'cds' in reg:# or 'mitranscriptome' in reg1 or 'trnascanse' in reg1 or 'snornabase' in reg1:
      test_ids_ex.append(id)
  test_ids = list(set(all_list)-set(test_ids_ex)-set(tjumirna_list))
  pos_ids.sort()
  neg_ids.sort()
  test_ids.sort()
  #print(len(pos_ids))
  #print(len(neg_ids))
  #print(len(tjumirna_list))
  #print(len(test_ids))
  return pos_ids, neg_ids, test_ids

def file2data(cancer_type, train_pos, train_neg, test_ids):
    mode_all = ['cadd', 'cna', 'css', 'mut', 'mut2', 'rna']
    tumor_set = {'Breast-AdenoCA':'Breast-AdenoCa', 'Eso-AdenoCA':'Eso-AdenoCa', 'Liver-HCC':'Liver-HCC', 'Ovary-AdenoCA':'Ovary-AdenoCa',
          'Stomach-AdenoCA':'Stomach-AdenoCa', 'Prost-AdenoCA':'Prost-AdenoCa', 'Head-SCC':'Head-SCC'}
    X_train = []
    X_test = []
    X = []
    for mode in mode_all:
      if mode != 'mut2':
        fea_one = f'{root_path}/code/Non-codingCode/ecancer_feature/{cancer_type}/{mode}.fea'
        df_one = pd.read_csv(fea_one, header=0, index_col=0, sep='\t')
        #训练数据X
        mat_train_pos = df_one.loc[train_pos, ::].values.astype(float)
        mat_train_neg = df_one.loc[train_neg, ::].values.astype(float)
        X_train.append(np.concatenate([mat_train_pos, mat_train_neg]))

        #测试数据X
        mat_test = df_one.loc[test_ids, ::].values.astype(float)
        X_test.append(mat_test)
      if mode == 'mut2':
        fea_one = f'{root_path}/code/Non-codingCode/ecancer_feature/{cancer_type}/{mode}.fea'
        df_one = pd.read_csv(fea_one, header=0, index_col=0, sep='\t')
        fea_sublist=['AAA_ref','AAC_ref','AAG_ref','AAT_ref','ACA_ref','ACC_ref','ACG_ref','ACT_ref','AGA_ref','AGC_ref',
                      'AGG_ref','ATA_ref','ATC_ref','ATG_ref','CAA_ref','CAC_ref','CAG_ref','CCA_ref','CCC_ref',
                      'CCG_ref','CGA_ref','CGC_ref','CTA_ref','CTC_ref','GAA_ref','GAC_ref','GCA_ref','GCC_ref',
                      'GGA_ref','GTA_ref','TAA_ref','TCA_ref']
        mat_train_pos = df_one.loc[train_pos, fea_sublist].values.astype(float)
        mat_train_neg = df_one.loc[train_neg, fea_sublist].values.astype(float)
        X_train.append(np.concatenate([mat_train_pos, mat_train_neg]))
        #测试数据X
        mat_test = df_one.loc[test_ids, fea_sublist].values.astype(float)
        X_test.append(mat_test)

    X_train=np.concatenate([X_train[0],X_train[1],X_train[2],X_train[3],X_train[4],X_train[5]],axis=1)
    Y_train = np.concatenate([np.ones((len(train_pos))), np.zeros((len(train_neg)))])

    X_test=np.concatenate([X_test[0],X_test[1],X_test[2],X_test[3],X_test[4],X_test[5]],axis=1)
    cla_X_train=pd.DataFrame(X_train)
    cla_X_train['class']=Y_train
    geneid=np.concatenate([train_pos,train_neg])
    cla_X_train['geneid']=geneid
    return X_train, Y_train, X_test, cla_X_train


n_input=82
class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.zeros(num_classes) #1行2列元素为0的张量
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class MultiHeadSelfAttention(nn.Module):
    dim_in: int  # input dimension
    dim_k: int  # key and query dimension
    dim_v: int  # value dimension
    num_heads: int  # number of heads, for each head, dim_* = dim_* // num_heads
    def __init__(self, dim_in, dim_k, dim_v, num_heads=2):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_k % num_heads == 0 and dim_v % num_heads == 0, "dim_k and dim_v must be multiple of num_heads"
        self.dim_in = dim_in
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.num_heads = num_heads
        self.linear_q = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_in, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_in, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k // num_heads)

    def forward(self, x):
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        nh = self.num_heads  # 2
        dk = self.dim_k // nh  # dim_k of each head 1
        dv = self.dim_v // nh  # dim_v of each head 1

        q = self.linear_q(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk) 5.reshape(16,5,2)
        k = self.linear_k(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        v = self.linear_v(x.reshape(batch, dim_in)).reshape(batch, nh, dv)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, self.dim_v)  # batch, n, dim_v
        return att

class Transformer(nn.Module):
    def __init__(self, n_input,):
        super(Transformer, self).__init__()
        self.n = 45  #46
        self.fc1 = nn.Linear(n_input, self.n)
        self.attn1 = MultiHeadSelfAttention(dim_in=self.n, dim_k=4, dim_v=8)
        self.fc5 = nn.Linear(8, 2)
        self.drop1 = nn.Dropout(0.004)

    def encoder(self, x):
        x1 = self.fc1(x)
        x1 = F.gelu(x1)
        x1 = self.drop1(x1)
        x2 = self.attn1(x1)
        x2 = F.gelu(x2)
        x5 = self.fc5(x2)
        x5 = F.softmax(x5)
        return x5

    def forward(self, x):
        z = self.encoder(x)
        return z

criteria = focal_loss(alpha=0.25, gamma=4.5) #alpha=0.25, gamma=2.5; alpha=0.25, gamma=4.5

#加载数据
def load_data(cancer_type, path):
    df = pd.read_csv(path, sep=',')
    list = df.columns.values.tolist()
    list.remove('geneid')
    list.remove('label')
    #ss=StandardScaler()
    ss=MinMaxScaler()
    #ss=RobustScaler()
    #ss=MaxAbsScaler()
    df[list] = ss.fit_transform(df[list])
    #joblib.dump(ss, f'{root_path}/code/Non-codingCode/ecancer_tier2/model/%s.tfscaler' % cancer_type)
    hg = df['geneid']
    targets = df['label']
    features = df.drop(['geneid', 'label'], axis=1)
    x = features.values
    y = targets.values
    y = y.astype(float)
    y = np.reshape(y, [len(y), 1])
    torch_data = MyDataset(x, y)
    return torch_data,hg

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)

#读取交叉验证的csv文件
def cv_data(cancer_type, batch_size):
  #print(batch_size)
  train_path = f'{root_path}/code/Non-codingCode/ecancer_tier2/result/%s_tfCV_2.csv' % cancer_type
  dataset, _ = load_data(cancer_type,train_path)
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True) #
  for batch_idx, (x, y) in enumerate(train_loader):
    x=np.array(x)
    y=np.array(y)
  return x,y

def fit_cv(cancer_type, Xs, y, k, b_plot=False):
    z = 0
    n = Xs.shape[0]
    tprs = []
    p = []
    x1_all = []
    x2_all = []
    mean_fpr = np.linspace(0, 1, 100)
    roc_auc = 0
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    all_roc = []
    pr=[]
    for j in range(1, k + 1):
        ix = assignments == j
        y_test = y[ix]
        y_train = y[~ix]
        X_train = Xs[~ix, :]
        X_test = Xs[ix, :]

        model = Transformer(n_input=n_input,)
        optimizer = Adam(model.parameters(), lr=0.006)
        X_test = torch.tensor(X_test)
        X_test = X_test.to(torch.float32)

        epoch1 = 24 #
        epoch2 = 51 #
        # 半监督模块
        ratio = 0.1  # 缺失值比例 0.1,0.35
        # 产生一个随机状态种子
        rng = np.random.RandomState(10)  #10
        YSemi_train = np.copy(y_train)
        YSemi_train = YSemi_train.squeeze()
        # print('YSemi_train>>>', YSemi_train.shape)
        # rng.rand()返回一个或一组服从“0~1”均匀分布的随机样本值
        YSemi_train[rng.rand(len(y_train)) < ratio] = -1


        # 把每次验证的训练数据X_train中的部分数据变为无标签的
        unlabeledX = X_train[YSemi_train == -1,:]
        unlabeledY = y_train[YSemi_train == -1]
        unlabeledY = unlabeledY.squeeze()
        idx = np.where(YSemi_train != -1)[0]
        labeledX = X_train[idx, :]
        labeledY = YSemi_train[idx]
        probThreshold = 0.05 #0.05
        # 外层循环是半监督的次数 不需要损失
        for epoch in range(epoch1):
            probThreshold += 0.01 #0.01
            loss_list=[]
            # 内层循环是transformer的学习
            for i in range(epoch2):
              total_loss = 0.
              optimizer.zero_grad()
              labeledX = torch.tensor(labeledX).to(torch.float32)
              labeledY = torch.tensor(labeledY).to(torch.int64)
              # print(labeledX.shape)
              unlabeledX = torch.tensor(unlabeledX).to(torch.float32)
              unlabeledY = torch.tensor(unlabeledY).to(torch.int64)
              preds_train = model(labeledX)
              loss = criteria(preds_train, labeledY)
              loss_list.append(loss.detach().numpy())
              total_loss += loss
              loss.backward()
              optimizer.step()
            # plt.plot(loss_list,label='Loss')
            # plt.show()

            preds_unlabeledY = model(unlabeledX)
            preds_unlabeledY_Prob = preds_unlabeledY[:, 1]
            preds_unlabeledY_Prob = preds_unlabeledY_Prob.unsqueeze(1)
            preds_unlabeledY_Prob = preds_unlabeledY_Prob.detach().numpy()
            labelidx = np.where(preds_unlabeledY_Prob > probThreshold)[0]
            #print(labelidx.shape)
            unlabelidx = np.where(preds_unlabeledY_Prob <= probThreshold)[0]
            labeledX = np.array(labeledX)
            labeledY = np.array(labeledY)
            unlabeledX = np.array(unlabeledX)
            unlabeledY = np.array(unlabeledY)
            labeledX = np.vstack((labeledX, unlabeledX[labelidx, :]))
            labeledY = np.hstack((labeledY, unlabeledY[labelidx]))
            unlabeledX = unlabeledX[unlabelidx, :]
            unlabeledY = unlabeledY[unlabelidx]
        
        model = joblib.load(f'{root_path}/code/Non-codingCode/ecancer_tier2/model/{cancer_type}/train/{cancer_type}_WGFormer_{j}.pkl')
        if cancer_type not in ['BRCA','STAD']:
            X_test=joblib.load(f'{root_path}/code/Non-codingCode/ecancer_tier2/model/{cancer_type}/train/{cancer_type}_X_test_{j}.pkl')
        preds_test = model(X_test)
        preds_te = preds_test[:, 1]
        preds_te = preds_te.unsqueeze(1)
        preds_te=preds_te.detach().numpy()
        fpr, tpr, thresholds = roc_curve(y_test, preds_te)

        # 记录最优门限
        roc_auc_1 = auc(fpr, tpr)
        print('Auroc：{}'.format(roc_auc_1))
        pr1=average_precision_score(y_test, preds_te)
        pr.append(pr1) #准备求平均的auprc
        optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        #print('门限：{}'.format(optimal_th))
        if roc_auc_1 > z:  # 准确率最大时，对应的门限值
            z = roc_auc_1
            m = optimal_th
            #joblib.dump(model, f'{root_path}/code/Non-codingCode/ecancer_tier2/model/%s_WGFormer_train.pkl' % cancer_type)
        del model
        for i in preds_te[y_test == 0]:
            x1_all.append(i)
        for i in preds_te[y_test == 1]:
            x2_all.append(i)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    tatistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')  # 秩和检验
    mean_auc = auc(mean_fpr, mean_tpr)
    #输出平均auprc
    sum=0
    for i in pr:
      sum += i
    mean_pr=sum/len(pr)
    print('Mean AUROC: {:.4f}'.format(mean_auc))
    print('Mean AUPRC : {:.4f}'.format(mean_pr))
    print('pvalue: {}'.format(pvalue))
    return m

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='eachcancer')
    parser.add_argument("-m", dest='mode', default="train", help="mode")
    parser.add_argument("-t", dest='type', default="STAD", help="cancer type")
    parser.add_argument("-o", dest='out', default="/content/drive/MyDrive/fulingya/", help="coding file")
    args = parser.parse_args()
    cancer_type=args.type

    df_tmp = pd.read_csv(f'{root_path}/code/Non-codingCode/chr_id.txt', header=0, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
    all_list = df_tmp.index.tolist()
    #print(len(all_list)) 146586

    train_pos, train_neg, test_ids = build_set(args.type, all_list)
    X_train, Y_train, X_test, cla_X_train = file2data(args.type, train_pos, train_neg, test_ids)
    #print(X_test.shape)
    #将交叉验证部分的训练数据保存为csv文件
    train_geneid = np.concatenate([train_pos, train_neg])
    df_train_geneid = pd.DataFrame(train_geneid)
    df_train_geneid.columns=['geneid']
    df_X_train = pd.DataFrame(X_train)
    df_Y_train = pd.DataFrame(Y_train)
    df_Y_train.columns = ['label']
    feature_train = pd.concat([df_train_geneid, df_X_train, df_Y_train], axis=1)
    feature_train.columns = ['geneid','CADD_mean','CADD_var',
            'amp_mean','amp_var','amp_freq','del_mean','del_var','del_freq','abs_mean','abs_var','abs_freq',
            'CSS_mean','CSS_var',
            'freq_Intron','freq_IGR','freq_RNA','freq_Missense_Mutation','freq_3UTR','freq_lincRNA',
            'freq_5Flank','freq_Silent','freq_5UTR','freq_Splice_Site','freq_Nonsense_Mutation',
            'freq_De_novo_Start_OutOfFrame','freq_Frame_Shift_Del','freq_In_Frame_Del',
            'freq_Frame_Shift_Ins','freq_De_novo_Start_InFrame','freq_Start_Codon_SNP',
            'freq_In_Frame_Ins','freq_Nonstop_Mutation','freq_Start_Codon_Del','freq_Stop_Codon_Del',
            'freq_Stop_Codon_Ins','freq_Start_Codon_Ins','freq_SNP','freq_DNP','freq_TNP','freq_DEL',
            'freq_INS','freq_ONP','sample_count_mean','sample_count_var','gc_mean','gc_var',
            'AAA_ref','AAC_ref','AAG_ref','AAT_ref','ACA_ref','ACC_ref','ACG_ref','ACT_ref','AGA_ref','AGC_ref',
            'AGG_ref','ATA_ref','ATC_ref','ATG_ref','CAA_ref','CAC_ref','CAG_ref','CCA_ref','CCC_ref',
            'CCG_ref','CGA_ref','CGC_ref','CTA_ref','CTC_ref','GAA_ref','GAC_ref','GCA_ref','GCC_ref',
            'GGA_ref','GTA_ref','TAA_ref','TCA_ref',
            'exp_mean','exp_var','rep_time','exp_CCLE','label']
    feature_train.to_csv(f'{root_path}/code/Non-codingCode/ecancer_tier2/result/{cancer_type}_tfCV_2.csv', index=False)

    #交叉验证
    x,y=cv_data(args.type,feature_train.shape[0])
    #print(x.shape,y.shape)
    fit_cv(args.type, x, y, 5, b_plot=False)

if __name__ == "__main__":
    main()
