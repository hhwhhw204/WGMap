import argparse
import os
import random
# from deepforest import CascadeForestClassifier
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from scipy import stats
from scipy.stats import fisher_exact
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
from sklearn.model_selection import train_test_split


def torch_seed(torch_seed):
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
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 找roc的最优阈值
def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def Find_Optimal_Cutoff_Pr(precision, recall, thresholds):
    difference = np.abs(precision - recall)
    min_difference_index = np.argmin(difference)
    # 选择对应的最佳阈值
    optimal_threshold = thresholds[min_difference_index]
    return optimal_threshold

def fisher_ex(a, b, c, d):
    _, pvalue = fisher_exact([[a, b], [c, d]], 'greater')
    if pvalue < 1e-256:
        pvalue = 1e-256
    # p1 = -math.log10(pvalue)
    return pvalue


def new_file2data(multiple):
    file = pd.read_csv(r'../data/train_data/pancan_tier1_49_1_' + str(multiple) + '.csv', sep='\t')
    pos_data = file.loc[(file['class']) == 1]
    pos_ids = pos_data['Hugo_Symbol'].values.tolist()
    pos_data = pos_data.drop(['Hugo_Symbol', 'class'], axis=1).values.astype(float)

    neg_data = file.loc[(file['class']) == 0]
    neg_ids = neg_data['Hugo_Symbol'].values.tolist()
    neg_data = neg_data.drop(['Hugo_Symbol', 'class'], axis=1).values.astype(float)

    X_train = np.concatenate([pos_data, neg_data])
    Y_train = np.concatenate([np.ones((len(pos_ids))), np.zeros((len(neg_ids)))])
    X_test = pd.read_csv(r'../../pancan_feature/pancan_fea.csv', sep=',')
    return X_train, Y_train, X_test


def fit_cv(X, Y, multiple, lr, batch_size, epoch):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=1, shuffle=True)
    print("X_train.shape=",X_train.shape)

    x1_all = []
    x2_all = []

    # 利用扩散模型实现数据扩增
    dataset = torch.Tensor(X_train).float()
    num_steps = 90  # 90
    betas = torch.linspace(-6, 6, num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, 0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]], 0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    assert alphas.shape == alphas_prod.shape == alphas_prod_p.shape == \
           alphas_bar_sqrt.shape == one_minus_alphas_bar_log.shape \
           == one_minus_alphas_bar_sqrt.shape


    # 编写拟合逆扩散过程高斯分布的模型
    class MLPDiffusion(nn.Module):
        def __init__(self, n_steps, size, num_units=182):  # 182
            super(MLPDiffusion, self).__init__()
            self.res1 = ResnetBlock(size, num_units)
            self.res2 = ResnetBlock(num_units, num_units)
            # self.attn = MultiHeadSelfAttention(size, num_units, num_units)
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

    # 编写训练的误差函数
    def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
        # 对任意时刻t进行采样计算loss
        batch_size = x_0.shape[0]
        # 对一个batchsize样本生成随机的时刻t
        t = torch.randint(0, n_steps, size=(batch_size // 2,))
        t = torch.cat([t, n_steps - 1 - t], dim=0)
        t = t.unsqueeze(-1)
        # x0的系数
        a = alphas_bar_sqrt[t]
        # eps的系数
        aml = one_minus_alphas_bar_sqrt[t]
        # 生成随机噪音eps
        e = torch.randn_like(x_0)
        # 构造模型的输入
        x = x_0 * a + e * aml
        # 送入模型，得到t时刻的随机噪声预测值
        output = model(x, t.squeeze(-1))
        # 与真实噪声一起计算误差，求平均值
        output = output[0]
        return (e - output).square().mean()

    # 编写逆扩散采样函数
    def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
        # 从x[T]恢复x[T-1]、x[T-2]|...x[0]
        cur_x = torch.randn(shape)
        x_seq = [cur_x]
        for i in reversed(range(n_steps)):
            cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
            x_seq.append(cur_x)
        return x_seq

    def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
        # 从x[T]采样t时刻的重构值
        t = torch.tensor([t])
        coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
        eps_theta = model(x, t)[0]
        mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
        z = torch.randn_like(x)
        sigma_t = betas[t].sqrt()
        sample = mean + sigma_t * z
        return (sample)

    # 开始训练模型，打印loss及中间重构效果
    print('Training model...')
    batch_size = batch_size  # 2
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    num_epoch = epoch
    model = MLPDiffusion(num_steps, dataset.shape[1])  # 输入是x和step
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 0.00005
    for t in range(num_epoch):
        for idx, batch_x in enumerate(dataloader):
            loss = diffusion_loss_fn(model, batch_x, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, num_steps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
        if (t == num_epoch - 1):
            x_seq = p_sample_loop(model, dataset.shape, num_steps, betas, one_minus_alphas_bar_sqrt)
            length = len(x_seq)
            X_train_augmentation = x_seq[length - 1]
    # print(X_train_augmentation.shape, type(X_train_augmentation))
    X_train_augmentation = X_train_augmentation.detach().numpy()
    y_train_augmentation = y_train

    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, '../model/minmax.scaler')

    scaler = joblib.load('../model/minmax.scaler')
    X_train_augmentation = scaler.fit_transform(X_train_augmentation)
    X_train_new = np.concatenate([X_train, X_train_augmentation], axis=0)
    y_train_new = np.concatenate([y_train, y_train_augmentation], axis=0)
    X_train_new = pd.DataFrame(X_train_new)
    X_train_new.fillna(method = 'ffill', inplace = True)
    X_train_new = np.array(X_train_new)

    model = xgb.XGBClassifier(random_state=0)
    model.fit(X_train_new, y_train_new)
    probas_ = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, probas_)

    # roc,pr
    roc_auc_1 = auc(fpr, tpr)
    pr1 = average_precision_score(y_test, probas_)

    precision, recall, thresholds = precision_recall_curve(y_test, probas_)
    thresholds = np.append(thresholds, 1)
    # 寻找精确度不低于0.75的阈值
    optimal_threshold = thresholds[precision >= 0.75][0]

    print('精确度和召回率指标的最优门限：{}'.format(optimal_threshold))
    joblib.dump(model, '../model/XGB.pkl')
    del model

    for j in probas_[y_test == 0]:
        x1_all.append(j)
    for j in probas_[y_test == 1]:
        x2_all.append(j)

    tatistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')  # 秩和检验
    print("auroc: {:.4f}, auprc: {:.4f}".format(roc_auc_1, pr1))
    print("len(x1_all)=", len(x1_all), "len(x2_all)=", len(x2_all), "tatistic=",tatistic)

    return roc_auc_1, pr1, pvalue, optimal_threshold  # 返回准确率最高时对应的门限值  作为预测阶段的阈值



def new_predict(X_test, multiple, m):
    Hugo_list = list(X_test['Hugo_Symbol'])
    X_test.drop(['Hugo_Symbol'], axis=1, inplace=True)

    scaler = joblib.load('../model/minmax.scaler')
    X_test = scaler.fit_transform(X_test)
    model = joblib.load('../model/XGB.pkl')
    probas_ = model.predict_proba(X_test)[:, 1]

    drug_num = 0
    drug_id = []
    drug_gene = []
    drug_prob = []
    probas_list = list(probas_)
    for i in range(len(probas_list)):
        if probas_list[i] >= m:
            drug_num += 1
            drug_id.append(i)
            drug_gene.append(Hugo_list[i])
            drug_prob.append(probas_list[i])
    print("druggable:", drug_num, "\n")
    return drug_num, drug_gene, drug_prob



def main():
    parser = argparse.ArgumentParser(description='coding pancan predict')
    parser.add_argument("-t", dest='type', default="Pancan", help="cancer type")
    parser.add_argument('-mul', dest='multiple', type=int, default=10)
    parser.add_argument('-l', dest='lr', type=float, default=0.008)
    parser.add_argument('-b', dest='batch_size', type=int, default=6)
    parser.add_argument('-e', dest='epoch', type=int, default=15)
    parser.add_argument("-o", dest='out', default="../result/")
    parser.add_argument("-ts", dest='ts', type=int,default=1)
    args = parser.parse_args()
    torch_seed(torch_seed=args.ts)

    X_train, Y_train, X_test = new_file2data(args.multiple)
    mean_auroc, mean_auprc, pvalue, m = fit_cv(X_train, Y_train, args.multiple, args.lr, args.batch_size, args.epoch)
    drug_num, drug_gene, drug_prob = new_predict(X_test, args.multiple, m)

    with open(args.out + "/tier1_pancan_druggable_genes.txt", 'w') as f:
        for i in range(drug_num):
            f.writelines(drug_gene[i] + "\t" + str(drug_prob[i]) + "\n")

if __name__ == "__main__":
    main()
