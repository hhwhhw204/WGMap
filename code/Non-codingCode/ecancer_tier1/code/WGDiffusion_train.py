import argparse
import sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import fisher_exact
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve,roc_curve,average_precision_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# from deepforest import CascadeForestClassifier
from sklearn import neighbors
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
import re
import os
import random

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
  pos_path = f'{root_path}/code/Non-codingCode/ecancer_tier1/data/{cancer_type}.csv'
  df_pos_noncds = pd.read_csv(pos_path, sep=',')
  pos_ids_noncds = df_pos_noncds['geneid'].values.tolist()
  pos_ids = list(set(pos_ids_noncds))
  #随机选择负样本 前提是排除可能的正样本
  tumor_list = ['BRCA', 'ESCA', 'LIHC', 'OV',
          'STAD', 'PRAD', 'HNSC']
  excludes = []
  for i in tumor_list:
    if i != cancer_type:
      path = f'{root_path}/code/Non-codingCode/ecancer_tier1/data/{i}.csv'
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
    if 'cds' not in reg_ffu and 'tjumirna' not in reg_ffu1:
      neg_ids_noncds.append(id) #非编码区其他负样本基因
  df_neg_ids_noncds = pd.DataFrame(neg_ids_noncds)
  df_fu = df_neg_ids_noncds.sample(n=len(pos_ids)*10,random_state=3,replace = False) #3
  neg_ids = df_fu[0].values.tolist()

  for id in all_list:
    tmps = re.split('::', id)
    gene = tmps[2]
    reg = tmps[0]
    reg1 = tmps[1]
    if 'cds' in reg:
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
    tumor_set = {'Breast-AdenoCA':'Breast-AdenoCa', 'Eso-AdenoCA':'Eso-AdenoCa', 'Liver-HCC':'Liver-HCC', 'Ovary-AdenoCA':'Ovary-AdenoCa',
          'Stomach-AdenoCA':'Stomach-AdenoCa', 'Prost-AdenoCA':'Prost-AdenoCa', 'Head-SCC':'Head-SCC'}
    mode_all = ['cadd', 'cna', 'css', 'mut', 'mut2', 'rna']
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

def fit_cv(cancer_type, Xs, y,cla_X_train, k,a,b,c,d, b_plot=False, method='0'):
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
  for i in range(1, k + 1):
    ix = assignments == i
    y_test = y[ix]
    y_train = y[~ix]
    X_train = Xs[~ix, :]
    X_test = Xs[ix, :]
    # print('X_train>>>',X_train.shape)
    # print('y_train>>>',y_train.shape)


    #利用扩散模型实现数据扩增
    #print(type(X_train))
    dataset = torch.Tensor(X_train).float()
    #确定超参数的值
    num_steps = 100 #100
    #制定每一步的beta
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5
    #计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
    alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
    ==one_minus_alphas_bar_sqrt.shape
    #print("all the same shape",betas.shape)
    #确定扩散过程任意时刻的采样值 可以基于x[0]得到任意时刻t的x[t]
    def q_x(x_0,t):
      noise = torch.randn_like(x_0)
      alphas_t = alphas_bar_sqrt[t]
      alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
      return (alphas_t * x_0 + alphas_1_m_t * noise) #在x[0]的基础上添加噪声

    #编写拟合逆扩散过程高斯分布的模型
    class MLPDiffusion(nn.Module):
      def __init__(self, n_steps, size, num_units=104): #104
        super(MLPDiffusion, self).__init__()
        self.res1 = ResnetBlock(size, num_units)
        self.res2 = ResnetBlock(num_units, num_units)
        self.attn = MultiHeadSelfAttention(size, num_units, num_units)
        self.emb = nn.Embedding(n_steps, num_units)
        self.linear = nn.Linear(num_units, size)
      def forward(self, x, t):
        x = self.res1(x)
        for i in range(2):
          x = self.res2(x)
          emb_t = self.emb(t)
          x += emb_t
          x = F.relu(x)
        y = self.linear(x)
        return y, x

    class ResnetBlock(nn.Module):
      def __init__(self, dim_in, dim_out):
        super().__init__()
        self.block1 = Block(dim_in, dim_out)
        self.block2 = Block(dim_out, dim_out)
        # self.attn = MultiHeadSelfAttention(dim_in, dim_out, dim_out)
        self.linear = nn.Linear(dim_in, dim_out)

      def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.linear(x)

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
        # x: tensor of shape (batch, n, dim_in)
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

    class Block(nn.Module):
      def __init__(self, dim_in, dim_out):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)
        # self.norm = nn.LayerNorm(dim_out)
        self.act = nn.GELU()

      def forward(self, x):
        x = self.linear(x)
        # x = self.norm(x)
        x = self.act(x)
        return x

    #编写训练的误差函数
    def diffusion_loss_fn(model,x_0,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,n_steps):
      #对任意时刻t进行采样计算loss
      batch_size = x_0.shape[0]
      #对一个batchsize样本生成随机的时刻t
      t = torch.randint(0,n_steps,size=(batch_size//2,))
      t = torch.cat([t,n_steps-1-t],dim=0)
      t = t.unsqueeze(-1)
      #x0的系数
      a = alphas_bar_sqrt[t]
      #eps的系数
      aml = one_minus_alphas_bar_sqrt[t]
      #生成随机噪音eps
      e = torch.randn_like(x_0)
      #构造模型的输入
      x = x_0*a+e*aml
      #送入模型，得到t时刻的随机噪声预测值
      output = model(x,t.squeeze(-1))
      #与真实噪声一起计算误差，求平均值
      output = output[0]
      return (e - output).square().mean()

    #编写逆扩散采样函数
    def p_sample_loop(model,shape,n_steps,betas,one_minus_alphas_bar_sqrt):
      #从x[T]恢复x[T-1]、x[T-2]|...x[0]
      cur_x = torch.randn(shape)
      x_seq = [cur_x]
      for i in reversed(range(n_steps)):
          cur_x = p_sample(model,cur_x,i,betas,one_minus_alphas_bar_sqrt)
          x_seq.append(cur_x)
      return x_seq

    def p_sample(model,x,t,betas,one_minus_alphas_bar_sqrt):
      #从x[T]采样t时刻的重构值
      t = torch.tensor([t])
      coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
      eps_theta = model(x,t)[0]
      mean = (1/(1-betas[t]).sqrt())*(x-(coeff*eps_theta))
      z = torch.randn_like(x)
      sigma_t = betas[t].sqrt()
      sample = mean + sigma_t * z
      return (sample)

    #开始训练模型，打印loss及中间重构效果
    #print('Training model...')
    batch_size = 2
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    num_epoch = 10 #10
    model = MLPDiffusion(num_steps,dataset.shape[1]) #输入是x和step
    optimizer = torch.optim.Adam(model.parameters(),lr=0.003) #0.003
    for t in range(num_epoch):
      loss_list = []
      for idx,batch_x in enumerate(dataloader):
        loss = diffusion_loss_fn(model,batch_x,alphas_bar_sqrt,one_minus_alphas_bar_sqrt,num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        loss = loss.detach().numpy()
        loss_list.append(loss)
      if(t==num_epoch-1):
        x_seq = p_sample_loop(model,dataset.shape,num_steps,betas,one_minus_alphas_bar_sqrt)
        length = len(x_seq)
        X_train_augmentation = x_seq[length - 1]
    # plt.plot(loss_list,'-^')
    # plt.show()
    # print(X_train_augmentation.shape,type(X_train_augmentation))
    X_train_augmentation = X_train_augmentation.detach().numpy()
    y_train_augmentation = y_train


    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, f'{root_path}/code/Non-codingCode/ecancer_tier1/model/{cancer_type}.xgbscaler')
    xtext = ix.nonzero()[0].tolist()
    df0 = cla_X_train.iloc[xtext,:]
    if method == 'XGB':
      model=joblib.load(f'{root_path}/code/Non-codingCode/ecancer_tier1/model/{cancer_type}/train/{cancer_type}_WGDiffusion_{i}.pkl')
      X_train_new = pd.read_csv(f'{root_path}/code/Non-codingCode/ecancer_tier1/data/{cancer_type}/{cancer_type}_X_train_new_{i}.csv',header=0)
      y_train_new = pd.read_csv(f'{root_path}/code/Non-codingCode/ecancer_tier1/data/{cancer_type}/{cancer_type}_y_train_new_{i}.csv',header=0)
      X_train_new = np.array(X_train_new)
      y_train_new = np.array(y_train_new)
      #model=xgb.XGBClassifier(random_state=0) #0
      #scaler=joblib.load(f'{root_path}/code/Non-codingCode/ecancer_tier1/model/{cancer_type}.xgbscaler')
      #X_train_augmentation = scaler.fit_transform(X_train_augmentation)
      #X_train_new=np.concatenate([X_train, X_train_augmentation],axis=0)
      #y_train_new=np.concatenate([y_train, y_train_augmentation],axis=0)



      model.fit(X_train_new, y_train_new)
      #model.fit(X_train, y_train)
      probas_ = model.predict_proba(X_test)[:, 1]
      fpr, tpr, thresholds = roc_curve(y_test, probas_)
      # print('X_train>>>',X_train)
      # print('probas_>>>',probas_)
      # print('y_test>>>',y_test)
      # 记录最优门限
      roc_auc_1 = auc(fpr, tpr)
      print('Auroc：{}'.format(roc_auc_1))
      pr1 = average_precision_score(y_test, probas_)
      pr.append(pr1) #准备求平均的auprc
      optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
      #print('门限：{}'.format(optimal_th))

      if roc_auc_1 > z: #准确率最大时，对应的门限值
        z = roc_auc_1
        m = optimal_th
        #joblib.dump(model, f'{root_path}/code/Non-codingCode/ecancer_tier1/model/{cancer_type}_WGDiffusion.pkl')
      del model

    for i in probas_[y_test==0]:
      x1_all.append(i)
    for i in probas_[y_test==1]:
      x2_all.append(i)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)

  mean_tpr = np.mean(tprs, axis=0)
  mean_tpr[-1] = 1.0

  # print(mean_pvalue)
  tatistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')#秩和检验
  mean_auc = auc(mean_fpr, mean_tpr)
  sum=0
  for i in pr:
    sum += i
  mean_pr=sum/len(pr)
  print('Mean AUROC: {:.4f}'.format(mean_auc))
  print('Mean AUPRC : {:.4f}'.format(mean_pr))
  print('pvalue: {}'.format(pvalue))
  return m #返回准确率最高时对应的门限值  作为预测阶段的阈值


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='eachcancer')
    parser.add_argument("-m", dest='mode', default="train", help="mode")
    parser.add_argument("-t", dest='type', default='STAD', help="cancer type")
    parser.add_argument("-o", dest='out', default="", help="coding file")
    args = parser.parse_args()
    cancer_type=args.type

    df_tmp = pd.read_csv(f'{root_path}/code/Non-codingCode/chr_id.txt', header=0, index_col=3, sep='\t', usecols=[0, 1, 2, 3])
    all_list = df_tmp.index.tolist()

    train_pos, train_neg, test_ids = build_set(args.type, all_list)
    X_train, Y_train, X_test, cla_X_train = file2data(args.type, train_pos, train_neg, test_ids)
    #print(X_train.shape)
    #print(Y_train.shape)
    #print(X_test.shape)
    #将交叉验证部分的训练数据保存为csv文件
    train_geneid = np.concatenate([train_pos, train_neg])
    df_train_geneid = pd.DataFrame(train_geneid)
    df_train_geneid.columns=['geneid']
    df_X_train = pd.DataFrame(X_train)
    df_Y_train = pd.DataFrame(Y_train)
    df_Y_train.columns = ['label']
    feature_train = pd.concat([df_train_geneid, df_X_train, df_Y_train], axis=1)
    feature_train.columns=['geneid','CADD_mean','CADD_var',
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
    #print(feature_train.shape)
    #feature_train.to_csv(rf'{root_path}/code/Non-codingCode/ecancer_tier1/result/%s_CV_1.csv' % args.type, index=False)

    if cancer_type == 'OV':
        k = 3
    elif cancer_type == 'HNSC':
        k = 4
    else:
        k = 5
    m = fit_cv(args.type,X_train,Y_train,cla_X_train,k,0,0,0,0, method='XGB')

if __name__ == "__main__":
    main()
