from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from math import sqrt
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, average_precision_score
from sklearn.preprocessing import MinMaxScaler
import warnings
import torch
from sklearn.metrics import auc
import os
from scipy.stats import fisher_exact
from scipy import stats
import joblib
warnings.filterwarnings('ignore')


def set_seed(seed):
    seed = seed
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def Find_Optimal_Cutoff_Pr(precision, recall, thresholds):
    difference = np.abs(precision - recall)
    min_difference_index = np.argmin(difference)
    optimal_threshold = thresholds[min_difference_index]
    return optimal_threshold



def file2data(multiple):
    file = pd.read_csv(r'../data/train_data/pancan_tier2_113_1_' + str(multiple) + '.csv', sep='\t')
    all_gene = pd.read_csv(r'../../pancan_feature/pancan_fea.csv', sep=',', index_col=0, header=0)
    all_gene_list = all_gene.index.values.tolist()

    pos_data = file.loc[(file['class']) == 1]
    pos_ids = pos_data['Hugo_Symbol'].values.tolist()
    pos_ids = [pos for pos in pos_ids if pos in all_gene_list]
    mat_train_pos = all_gene.loc[pos_ids, ::].values.astype(float)

    neg_data = file.loc[(file['class']) == 0]
    neg_ids = neg_data['Hugo_Symbol'].values.tolist()
    neg_ids = [neg for neg in neg_ids if neg in all_gene_list]
    mat_train_neg = all_gene.loc[neg_ids, ::].values.astype(float)

    X_train = np.concatenate([mat_train_pos, mat_train_neg])
    Y_train = np.concatenate([np.ones((len(pos_ids))), np.zeros((len(neg_ids)))])

    X_test = pd.read_csv(r'../../pancan_feature/pancan_fea.csv', sep=',')
    return pos_ids, neg_ids, X_train, Y_train, X_test



n_input = 82
cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")


class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=2, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        else:
            self.alpha = torch.zeros(num_classes) 
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  
        preds_softmax = torch.exp(preds_logsoft)  
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss



class MultiHeadSelfAttention(nn.Module):
    dim_in: int  
    dim_k: int  
    dim_v: int 
    num_heads: int 

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
        nh = self.num_heads  
        dk = self.dim_k // nh 
        dv = self.dim_v // nh 
        q = self.linear_q(x.reshape(batch, dim_in)).reshape(batch, nh, dk) 
        k = self.linear_k(x.reshape(batch, dim_in)).reshape(batch, nh, dk) 
        v = self.linear_v(x.reshape(batch, dim_in)).reshape(batch, nh, dv)  
        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact 
        dist = torch.softmax(dist, dim=-1) 
        att = torch.matmul(dist, v)  
        att = att.transpose(1, 2).reshape(batch, self.dim_v)
        return att


class Transformer(nn.Module):
    def __init__(self, n_input, ):
        super(Transformer, self).__init__()
        self.n = 24 
        self.fc1 = nn.Linear(n_input, self.n)
        self.attn1 = MultiHeadSelfAttention(dim_in=self.n, dim_k=4, dim_v=18) 
        self.fc5 = nn.Linear(18, 2)
        self.drop1 = nn.Dropout(0.05)

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


def load_data(path, multiple):
    df = pd.read_csv(path, sep=',')
    list = df.columns.values.tolist()
    list.remove('geneid')
    list.remove('label')
    ss = MinMaxScaler()
    df[list] = ss.fit_transform(df[list])
    joblib.dump(ss, '../model/train/trans_predruggable.scaler')
    hg = df['geneid']
    targets = df['label']
    features = df.drop(['geneid', 'label'], axis=1)
    x = features.values
    y = targets.values
    y = y.astype(float)
    y = np.reshape(y, [len(y), 1])
    torch_data = MyDataset(x, y)
    return torch_data, hg


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


def cv_data(batch_size, multiple):
    train_path = r'../data/trans_pre_CV.csv'
    dataset, _ = load_data(train_path, multiple)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)  # 这里shuffle不能设置为True
    for batch_idx, (x, y) in enumerate(train_loader):
        x = np.array(x)
        y = np.array(y)
    return x, y


def fit_cv(Xs, y, k, lr, e1, e2, multiple, ratio):
    z = 0
    n = Xs.shape[0]
    tprs = []
    x1_all = []
    x2_all = []
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]
    mean_fpr = np.linspace(0, 1, 100)
    pr = []
    roc_list = []

    for i in range(1, k + 1):
        ix = assignments == i
        y_test = y[ix]
        y_train = y[~ix]
        X_train = Xs[~ix, :]
        X_test = Xs[ix, :]

        model = Transformer(n_input=n_input, )
        optimizer = Adam(model.parameters(), lr=lr)
        X_test = torch.tensor(X_test).to(torch.float32)
        criteria = focal_loss(alpha=0.25, gamma=4)
        ratio = ratio
        rng = np.random.RandomState(10) 
        YSemi_train = np.copy(y_train)
        YSemi_train = YSemi_train.squeeze()
        YSemi_train[rng.rand(len(y_train)) < ratio] = -1
        unlabeledX = X_train[YSemi_train == -1, :]
        unlabeledY = y_train[YSemi_train == -1]
        unlabeledY = unlabeledY.squeeze()
        idx = np.where(YSemi_train != -1)[0]
        labeledX = X_train[idx, :]
        labeledY = YSemi_train[idx]
        probThreshold = 0.05  

        for epoch in range(e1):
            probThreshold += 0.01
            loss_list = []
            # 内层循环是transformer的学习
            for i in range(e2):
                total_loss = 0.
                optimizer.zero_grad()
                labeledX = torch.tensor(labeledX).to(torch.float32)
                labeledY = torch.tensor(labeledY).to(torch.int64)
                unlabeledX = torch.tensor(unlabeledX).to(torch.float32)
                unlabeledY = torch.tensor(unlabeledY).to(torch.int64)
                preds_train = model(labeledX)
                loss = criteria(preds_train, labeledY)
                loss_list.append(loss.detach().numpy())
                total_loss += loss
                loss.backward()
                optimizer.step()
            preds_unlabeledY = model(unlabeledX)
            preds_unlabeledY_Prob = preds_unlabeledY[:, 1]
            preds_unlabeledY_Prob = preds_unlabeledY_Prob.unsqueeze(1)
            preds_unlabeledY_Prob = preds_unlabeledY_Prob.detach().numpy()
            labelidx = np.where(preds_unlabeledY_Prob > probThreshold)[0]
            unlabelidx = np.where(preds_unlabeledY_Prob <= probThreshold)[0]
            labeledX = np.array(labeledX)
            labeledY = np.array(labeledY)
            unlabeledX = np.array(unlabeledX)
            unlabeledY = np.array(unlabeledY)
            labeledX = np.vstack((labeledX, unlabeledX[labelidx, :]))
            labeledY = np.hstack((labeledY, unlabeledY[labelidx]))
            unlabeledX = unlabeledX[unlabelidx, :]
            unlabeledY = unlabeledY[unlabelidx]

        preds_test = model(X_test)
        preds_te = preds_test[:, 1]
        preds_te = preds_te.unsqueeze(1)
        preds_te = preds_te.detach().numpy()
        fpr, tpr, thresholds = roc_curve(y_test, preds_te)

        roc_auc_1 = auc(fpr, tpr)
        print('auroc:{}'.format(roc_auc_1))
        roc_list.append(roc_auc_1)

        pr1 = average_precision_score(y_test, preds_te)
        pr.append(pr1)  

        optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        print('门限：{}'.format(optimal_th))

        if roc_auc_1 > z:
            z = roc_auc_1
            m = optimal_th
            joblib.dump(model, '../model/train/trans_pre.pkl')
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

    sum = 0
    for i in pr:
        sum += i
    mean_pr = sum / len(pr)
    print('mean_auprc : {:.4f}'.format(mean_pr))

    sum = 0
    for i in roc_list:
        sum += i
    mean_auroc = sum / len(roc_list)

    tatistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')  # 秩和检验
    print("my mean_auroc: {:.4f}, p-value = {:.8e}".format(mean_auroc, pvalue))
    return mean_auroc, mean_pr, pvalue, m



def main():
    parser = argparse.ArgumentParser(description='coding_pancan_tier2')
    parser.add_argument('-m', dest='mode', type=str, default='predict')
    parser.add_argument('-mul', dest='multiple', type=int, default=10)
    parser.add_argument('-l', dest='lr', type=float, default=0.0005)
    parser.add_argument('-e1', dest='epoch1', type=int, default=15)
    parser.add_argument('-e2', dest='epoch2', type=int, default=45)
    parser.add_argument('-ratio', dest='ratio', type=float, default=0.3)
    parser.add_argument("-o", dest='out', default="../result/", help="coding file")
    parser.add_argument('-ts', dest='tseed', type=int, default=1)
    args = parser.parse_args()
    set_seed(seed=args.tseed)

    train_pos, train_neg, X_train, Y_train, X_test = file2data(args.multiple)
    train_geneid = np.concatenate([train_pos, train_neg])
    df_train_geneid = pd.DataFrame(train_geneid)
    df_train_geneid.columns = ['geneid']
    df_X_train = pd.DataFrame(X_train)
    df_Y_train = pd.DataFrame(Y_train)
    df_Y_train.columns = ['label']
    feature_train = pd.concat([df_train_geneid, df_X_train, df_Y_train], axis=1)
    print(feature_train.shape)
    feature_train.to_csv(r'../data/trans_pre_CV.csv', index=False)

    batch_size = feature_train.shape[0]
    x, y = cv_data(batch_size, args.multiple)
    mean_auc, mean_pr, pvalue, m = fit_cv(x, y, 5, args.lr, args.epoch1, args.epoch2, args.multiple, args.ratio)


if __name__ == "__main__":
    main()
