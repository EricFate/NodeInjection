from pygcn.utils import normalize, count_acc, sparse_mx_to_torch_sparse_tensor, convert_to_coo
from torch.nn import functional as F
import torch as t
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
import pickle as pk


class AFGSM:
    def __init__(self, args, real_model, ada=True):
        self.ada = ada
        self.args = args
        self.real_model = real_model

    def attack(self, model, X, A: sp.csr_matrix, labels, train_idx):
        origin_dim = A.shape[0]
        num_attack = 500
        pre_attack = np.zeros(num_attack)
        self.eval_real(X, A, labels, train_idx)
        A_hat = normalize(A, symmetric=True)
        for i in range(self.args.num_adv):
            print('----------------  adding node %d -------------------------' % (i + 1))
            D_hat = np.sum(A, axis=1) + 1
            D_inv_sqrt = np.power(D_hat, -0.5)
            c_new, dif = self.retrain_model(model, X, A_hat, labels, train_idx)
            d = 100
            attack_idx = np.argpartition(dif, -num_attack)[-num_attack:]
            e = np.random.choice(X.shape[0], d)
            x = np.random.randn(X.shape[1])
            W = model.W.cpu().detach().numpy()
            # num_attack = np.sum(np.sort(pre_attack) == np.sort(attack_idx))
            print('# of same attack node :%d' % len(set(pre_attack) & set(attack_idx)))
            pre_attack = attack_idx
            done = False
            count = 0
            # while not done:
                # count += 1
                # print('iteration %d' % count)
            dx = self.compute_dx(d + 1, D_inv_sqrt, A_hat, e, attack_idx, W, c_new, labels)
            x = 0.5 - 0.5 * np.sign(dx)
            de = self.compute_de(d + 1, D_inv_sqrt, X, A_hat, x, e, attack_idx, W, c_new, labels)
            e = np.argpartition(de, d)[:d]
                # e_.sort()
                # done = np.all(e_ == e)
                # e = e_
            X = np.concatenate([X, x[np.newaxis, :]], axis=0)
            new_dim = A.shape[0] + 1
            lil_A = A.tolil()
            lil_A.resize(new_dim, new_dim)
            lil_A[e, new_dim - 1] = 1
            lil_A[new_dim - 1, e] = 1
            A = lil_A.tocsr()
            A_hat = normalize(A, symmetric=True)
            self.eval_model(model, X, A_hat, labels, train_idx)
            self.eval_real(X, A, labels, train_idx)
        new_adj = A[origin_dim:, :]
        new_feat = X[origin_dim:, :]
        np.save('feature.npy', new_feat)
        with open('adj.pkl', 'wb') as f:
            pk.dump(f, new_adj)

    def compute_de(self, d, D_inv_sqrt, X, A_hat, x: np.ndarray, e, train_idx, W, c_new, label):
        de = 0.
        for i in train_idx:
            c_n = c_new[i]
            c = label[i]
            dW = W[:, c] - W[:, c_n]

            delta = (A_hat[:, i].multiply(D_inv_sqrt) * np.sum(x * dW) / D_inv_sqrt[i].item()).A.reshape(-1)
            if i in e:
                delta += D_inv_sqrt.A.reshape(-1) * X.dot(dW) / np.sqrt(d)
            de += delta * D_inv_sqrt[i].item()
        return de

    def compute_dx(self, d, D_inv_sqrt, A_hat, e, train_idx, W, c_new, label):
        dx = 0
        for i in train_idx:
            c_n = c_new[i]
            c = label[i]
            dm = A_hat[e, i].T.dot(D_inv_sqrt[e]).item()
            if i in e:
                dm += D_inv_sqrt[i].item() / d
            dW = W[:, c] - W[:, c_n]
            dx += dW
        return dx

    def retrain_model(self, model, X, A, labels, train_idx):
        print('----------------- retraining -------------------')
        # model.reset_parameters()
        optim = t.optim.Adam(model.parameters(), lr=self.args.lr)
        X = t.from_numpy(X).float().to(self.args.device)
        A = sparse_mx_to_torch_sparse_tensor(A).to(self.args.device)
        labels = t.from_numpy(labels).long().to(self.args.device)
        for i in tqdm(range(self.args.retrain_epoch), total=self.args.retrain_epoch, desc='retraining'):
            logits = model(X, A)
            loss = F.cross_entropy(logits[train_idx], labels[train_idx])
            acc_train = count_acc(logits[train_idx], labels[train_idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
            # print('Epoch: {:04d}'.format(i + 1),
            #       'loss_train: {:.4f}'.format(loss.item()),
            #       'acc_train: {:.4f}'.format(acc_train))
        print('----------------- retraining finished -------------------')
        with t.no_grad():
            logits = model(X, A)
            train_logits = logits[train_idx]
            train_labels = labels[train_idx]
            acc_train = count_acc(train_logits, train_labels)
            print('training acc: %.4f' % acc_train)
            train_labels_onehot = t.zeros_like(train_logits) \
                .to(self.args.device) \
                .scatter_(1, train_labels.view(-1, 1), 1)
            l_val = train_logits.gather(1, train_labels.view(-1, 1)).clone()
            c_val, c_new = t.max(train_logits - 1e6 * train_labels_onehot, dim=1)
            dif = l_val.squeeze() - c_val
            return c_new, dif.cpu().detach().numpy()

    def eval_model(self, model, X, A, labels, train_idx):
        X = t.from_numpy(X).float().to(self.args.device)
        A = sparse_mx_to_torch_sparse_tensor(A).to(self.args.device)
        labels = t.from_numpy(labels).long().to(self.args.device)
        with t.no_grad():
            logits = model(X, A)
            acc_train = count_acc(logits[train_idx], labels[train_idx])
            print('training acc: %.4f' % acc_train)

    def eval_real(self, X, A, labels, train_idx):
        edge_index, edge_weight = convert_to_coo(A)
        features = t.from_numpy(X).cuda().float()
        edge_index = t.from_numpy(edge_index).cuda().long()
        edge_weight = t.from_numpy(edge_weight).cuda().float()
        labels = t.from_numpy(labels).cuda().long()
        with t.no_grad():
            logits = self.real_model(features, edge_index, edge_weight)
            acc_train = count_acc(logits[train_idx], labels[train_idx])
            print('real model acc: %.4f' % acc_train)
