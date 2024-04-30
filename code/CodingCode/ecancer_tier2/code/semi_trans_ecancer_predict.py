from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from math import sqrt
import numpy as np
import random
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler  # 标准化工具
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
import warnings
import torch
from sklearn.metrics import auc
import os
import joblib
torch.set_printoptions(threshold=np.inf)


def set_seed(torch_seed):
    warnings.filterwarnings('ignore')
    seed = torch_seed  
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def new_file2data(cancer_type, multiple):
    file = pd.read_csv(
        r'../data/train_data/{}_1_{}.csv'.format(cancer_type, str(multiple)))

    pos_data = file.loc[(file['class']) == 1.0]
    pos_ids = pos_data['Hugo_Symbol'].values.tolist()
    pos_data = pos_data.drop(['Hugo_Symbol', 'class'], axis=1).values.astype(float)

    neg_data = file.loc[(file['class']) == 0.0]
    neg_ids = neg_data['Hugo_Symbol'].values.tolist()[: multiple * len(pos_ids)]
    neg_data = neg_data.drop(['Hugo_Symbol', 'class'], axis=1).values.astype(float)

    X_train = np.concatenate([pos_data, neg_data])
    Y_train = np.concatenate([np.ones((len(pos_ids))), np.zeros((len(neg_ids)))])

    X_test = pd.read_csv("../../ecancer_feature/{}_fea.csv".format(cancer_type), sep=',')
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

        q = self.linear_q(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        k = self.linear_k(x.reshape(batch, dim_in)).reshape(batch, nh, dk)  # (batch, nh, n, dk)
        v = self.linear_v(x.reshape(batch, dim_in)).reshape(batch, nh, dv)  # (batch, nh, n, dv)

        dist = torch.matmul(q, k.transpose(1, 2)) * self._norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # batch, nh, n, n

        att = torch.matmul(dist, v)  # batch, nh, n, dv
        att = att.transpose(1, 2).reshape(batch, self.dim_v)  # batch, n, dim_v
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


def load_data(path, multiple,cancer):
    df = pd.read_csv(path, sep=',')
    list = df.columns.values.tolist()
    list.remove('geneid')
    list.remove('label')

    ss = MinMaxScaler()
    df[list] = ss.fit_transform(df[list])
    joblib.dump(ss, '../model/{}/trans_predruggable.scaler'.format(cancer))
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


def cv_data(cancer,batch_size, multiple):
    train_path = r'../data/train_data/trans_pre_CV.csv'
    dataset, _ = load_data(train_path, multiple,cancer)
    if cancer == 'ov':
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    for batch_idx, (x, y) in enumerate(train_loader):
        x = np.array(x)
        y = np.array(y)
    return x, y


def fit_cv(X, Y, cancer, lr, epoch1, epoch2, multiple, test_size, shuffle_seed):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=shuffle_seed,
                                                        shuffle=True)
    x1_all = []
    x2_all = []
    model = Transformer(n_input=n_input, )
    optimizer = Adam(model.parameters(), lr=lr)
    X_test = torch.tensor(X_test)
    X_test = X_test.to(torch.float32)

    ratio = 0.3
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
    probThreshold = 0.05  # 0.05

    criteria = focal_loss(alpha=0.25, gamma=4.5)
    for epoch in range(epoch1):
        probThreshold += 0.01
        loss_list = []
        for j in range(epoch2):
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
    pr1 = average_precision_score(y_test, preds_te)

    precision, recall, thresholds = precision_recall_curve(y_test, preds_te)
    thresholds = np.append(thresholds, 1)
    optimal_threshold = thresholds[precision >= 0.90][0]

    print('Optimal thresholds for accuracy and recall metrics:{}'.format(optimal_threshold))
    joblib.dump(optimal_threshold, '../model/{}/optimal_threshold.pkl'.format(cancer))
    joblib.dump(model, '../model/{}/{}_trans.pkl'.format(cancer,cancer))
    del model

    for j in preds_te[y_test == 0]:
        x1_all.append(j)
    for j in preds_te[y_test == 1]:
        x2_all.append(j)

    pvalue = 0.0
    return roc_auc_1, pr1, pvalue, optimal_threshold


def predict(X_test, cancer, multiple):
    Hugo_list = list(X_test['Hugo_Symbol'])
    X_test.drop(['Hugo_Symbol'], axis=1, inplace=True)

    m = joblib.load('../model/{}/optimal_threshold.pkl'.format(cancer))
    ss = joblib.load('../model/{}/trans_predruggable.scaler'.format(cancer))
    X_test = ss.transform(X_test)

    X_test = torch.tensor(X_test).to(torch.float32)
    model = joblib.load('../model/{}/{}_trans.pkl'.format(cancer,cancer))
    probas_ = model(X_test)[:, 1]

    drug_num = 0
    drug_gene = []
    drug_prob = []
    probas_list = list(probas_)
    for i in range(len(probas_list)):
        if probas_list[i] >= m:
            drug_num += 1
            drug_gene.append(Hugo_list[i])
            drug_prob.append(probas_list[i].item())
    print("druggable:", drug_num, "\n")
    return drug_num, drug_gene, drug_prob


def main():
    parser = argparse.ArgumentParser(description='cds_pre_noncds')
    parser.add_argument("-m", dest='mode', default="pred", help="mode")
    parser.add_argument("-t", dest='cancer_type', default="prad", help="cancer type")
    parser.add_argument("-o", dest='out', default="../result/", help="coding file")
    parser.add_argument('-mul', dest='multiple', type=int, default=10)
    parser.add_argument('-l', dest='lr', type=float, default=0.0018)
    parser.add_argument('-e1', dest='epoch1', type=int, default=5)
    parser.add_argument('-e2', dest='epoch2', type=int, default=25)
    parser.add_argument('-ts', dest='torch_seed', type=int, default=1)
    parser.add_argument("-test_size", dest='test_size', type=int, default=0.1)
    parser.add_argument("-shuffle_seed", dest='shuffle_seed', type=int, default=3)
    args = parser.parse_args()
    set_seed(args.torch_seed)

    train_pos, train_neg, X_train, Y_train, X_test = new_file2data(args.cancer_type, args.multiple)
    train_geneid = np.concatenate([train_pos, train_neg])
    df_train_geneid = pd.DataFrame(train_geneid)
    df_train_geneid.columns = ['geneid']
    df_X_train = pd.DataFrame(X_train)
    df_Y_train = pd.DataFrame(Y_train)
    df_Y_train.columns = ['label']
    feature_train = pd.concat([df_train_geneid, df_X_train, df_Y_train], axis=1)
    feature_train.to_csv(r'../data/train_data/trans_pre_CV.csv', index=False)

    batch_size = feature_train.shape[0]
    x, y = cv_data(args.cancer_type, batch_size, args.multiple)

    mean_auroc, mean_auprc, pvalue, m  = fit_cv(x, y, args.cancer_type, args.lr, args.epoch1, args.epoch2, args.multiple,
                                   args.test_size, args.shuffle_seed)

    drug_num, drug_gene, drug_prob = predict(X_test, args.cancer_type, args.multiple)


    with open(args.out + "/tier2_{}_druggable_genes.txt".format(args.cancer_type), 'w') as f:
        for i in range(drug_num):
            f.writelines(drug_gene[i] + "\t" + str(drug_prob[i]) + "\n")


if __name__ == "__main__":
    main()
