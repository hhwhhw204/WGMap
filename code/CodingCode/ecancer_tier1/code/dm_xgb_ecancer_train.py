import argparse
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn import preprocessing
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
warnings.filterwarnings('ignore')


def set_seed(torch_seed):
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


optimal_th = 0.5
our_threshold = 0.05


def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def load_data(cancer_type, multiple):
    all_train_data = pd.read_csv(r'../data/train_data/{}_1_{}.csv'.format(cancer_type, multiple),
                                 sep=',')
    train_pos = all_train_data.loc[(all_train_data['class'] == 1.0)]
    train_pos_genes = train_pos['Hugo_Symbol'].values.tolist()
    print("正样本数量=", len(train_pos_genes))
    train_pos.drop(['Hugo_Symbol', 'class'], axis=1, inplace=True)
    train_neg = all_train_data.loc[(all_train_data['class'] == 0.0)]
    train_neg.drop(['Hugo_Symbol', 'class'], axis=1, inplace=True)
    X_train = np.concatenate([train_pos, train_neg])
    Y_train = np.concatenate([np.ones((len(train_pos))), np.zeros((len(train_neg)))])
    return X_train, Y_train


def fit_cv(Xs, y, cancer_type, k, multiple, lr, bs, epoch, unit=182):
    z = 0
    n = Xs.shape[0]
    tprs = []
    x1_all = []
    x2_all = []
    assignments = np.array((n // k + 1) * list(range(1, k + 1)))
    assignments = assignments[:n]
    mean_fpr = np.linspace(0, 1, 100)
    roc_list = []
    pr = []


    for i in range(1, k + 1):
        ix = assignments == i
        y_test = y[ix]
        y_train = y[~ix]
        X_train = Xs[~ix, :]
        X_test = Xs[ix, :]

        dataset = torch.Tensor(X_train)
        num_steps = 90
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

        class MLPDiffusion(nn.Module):
            def __init__(self, n_steps, size, num_units=unit):
                super(MLPDiffusion, self).__init__()
                self.res1 = ResnetBlock(size, num_units)
                self.res2 = ResnetBlock(num_units, num_units)
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
                self.linear = nn.Linear(dim_in, dim_out)

            def forward(self, x):
                h = self.block1(x)  
                h = self.block2(h)  
                return h + self.linear(x)

        class Block(nn.Module):
            def __init__(self, dim_in, dim_out):
                super().__init__()
                dim_in = int(dim_in)
                dim_out = int(dim_out)
                self.linear = nn.Linear(dim_in, dim_out)
                self.act = nn.GELU()

            def forward(self, x):
                x = self.linear(x)
                x = self.act(x)
                return x

        def diffusion_loss_fn(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
            batch_size = x_0.shape[0]
            t = torch.randint(0, n_steps, size=(batch_size // 2,))
            t = torch.cat([t, n_steps - 1 - t], dim=0)
            t = t.unsqueeze(-1)
            a = alphas_bar_sqrt[t]
            aml = one_minus_alphas_bar_sqrt[t]
            e = torch.randn_like(x_0)
            x = x_0 * a + e * aml
            output = model(x, t.squeeze(-1))
            output = output[0]
            return (e - output).square().mean()

        def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt):
            cur_x = torch.randn(shape)
            x_seq = [cur_x]
            for i in reversed(range(n_steps)):
                cur_x = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt)
                x_seq.append(cur_x)
            return x_seq

        def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt):
            t = torch.tensor([t])
            coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
            eps_theta = model(x, t)[0]
            mean = (1 / (1 - betas[t]).sqrt()) * (x - (coeff * eps_theta))
            z = torch.randn_like(x)
            sigma_t = betas[t].sqrt()
            sample = mean + sigma_t * z
            return (sample)

        print('Training model...')
        batch_size = bs  # 2
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        num_epoch = epoch
        model = MLPDiffusion(num_steps, dataset.shape[1]) 
        optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
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

        X_train_augmentation = X_train_augmentation.detach().numpy()
        y_train_augmentation = y_train

        scaler = preprocessing.MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train_augmentation = scaler.fit_transform(X_train_augmentation)
        X_train_new = np.concatenate([X_train, X_train_augmentation], axis=0)
        y_train_new = np.concatenate([y_train, y_train_augmentation], axis=0)
        X_train_new = pd.DataFrame(X_train_new)
        X_train_new.fillna(method = 'ffill', inplace = True)
        X_train_new = np.array(X_train_new)

        model = xgb.XGBClassifier(random_state=10)
        model.fit(X_train_new, y_train_new)
        probas_ = model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, probas_)
        roc_auc = auc(fpr, tpr)
        print('auroc:{}'.format(roc_auc))
        roc_list.append(roc_auc)

        pr1 = average_precision_score(y_test, probas_)
        pr.append(pr1)
        optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
        print('optimal_th:{}'.format(optimal_th))
        
        joblib.dump(X_test,'../model/{}/x_{}fold.pkl'.format(cancer_type,i))
        joblib.dump(y_test,'../model/{}/y_{}fold.pkl'.format(cancer_type,i))
        joblib.dump(model, '../model/{}/XGB_{}fold.pkl'.format(cancer_type,i))
        
        if roc_auc > z:
            z = roc_auc
            m = optimal_th
            joblib.dump(model, '../model/{}/XGB_pre_druggable_{}.pkl'.format(cancer_type,cancer_type))
            joblib.dump(scaler, '../model/{}/minmax_best.scaler'.format(cancer_type))
        del model

        x1_now, x2_now = [], []
        for j in probas_[y_test == 0]:
            x1_all.append(j)
            x1_now.append(j)
        for j in probas_[y_test == 1]:
            x2_all.append(j)
            x2_now.append(j)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

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

    tatistic, pvalue = stats.mannwhitneyu(x1_all, x2_all, use_continuity=True, alternative='two-sided')
    print("my mean_auroc: {:.8f}, p-value = {:.8f}".format(mean_auroc, pvalue))
    return m, mean_auroc, mean_pr, pvalue



def main():
    parser = argparse.ArgumentParser(description='coding_ecancer_tier1')
    parser.add_argument("-t", dest='type', default="prad", help="cancer type")
    parser.add_argument("-m", dest='mode', default="predict")
    parser.add_argument('-mul', dest='multiple', type=int, default=10)
    parser.add_argument('-l', dest='lr', type=float, default=0.0008)
    parser.add_argument('-b', dest='batch_size', type=int, default=8)
    parser.add_argument('-e', dest='epoch', type=int, default=12)
    parser.add_argument("-o", dest='out', default="../result/", help="coding data")
    parser.add_argument("-ts", dest='torch_seed', type=int, default=0)
    args = parser.parse_args()
    set_seed(args.torch_seed)

    if args.mode != "predict":
        X_train, Y_train = load_data(args.type, args.multiple)
        fit_cv(X_train, Y_train, args.type, 5, args.multiple, args.lr, args.batch_size, args.epoch)
    else:
        roc_list = []
        pr_list =[]
        for i in range(1,6):
            x = joblib.load('../model/{}/x_{}fold.pkl'.format(args.type,i))
            y = joblib.load('../model/{}/y_{}fold.pkl'.format(args.type,i))
            xgb = joblib.load('../model/{}/XGB_{}fold.pkl'.format(args.type,i))

            probas_ = xgb.predict_proba(x)[:, 1]
            fpr, tpr, thresholds = roc_curve(y, probas_)
            roc_auc_1 = auc(fpr, tpr)
            print("auroc=",roc_auc_1)
            
            roc_list.append(roc_auc_1)
            pr1 = average_precision_score(y, probas_)
            pr_list.append(pr1)
            print("auprc=",pr1)

        sum = 0
        for i in roc_list:
            sum += i
        mean_auroc = sum / len(roc_list)
        print('mean_auroc : {:.4f}'.format(mean_auroc))

        sum = 0
        for i in pr_list:
            sum += i
        mean_pr = sum / len(pr_list)
        print('mean_auprc : {:.4f}'.format(mean_pr))



if __name__ == "__main__":
    main()
