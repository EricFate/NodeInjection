import torch as t
import torch.optim as optim

from pygcn.utils import count_acc
from scipy.sparse import coo_matrix
import numpy as np
import pickle as pk
from torch.nn import functional as F
from tqdm import tqdm
import os


class GradAttack:
    def __init__(self, args, real_model, ada=True):
        self.ada = ada
        self.args = args
        self.real_model = real_model
        self.normalize = args.normalize
        self.dirname = 'norm_{}_targ_{}_ep{}_api{}'.format(self.normalize, self.args.targeted,args.adv_feature_epoch,args.adv_per_iter)
        path = os.path.join('result', self.dirname)
        if not os.path.exists(path):
            os.mkdir(path)

    def attack(self, X, edge_index, edge_weight, labels, attack_idx):
        # self.eval_real(X, edge_index, edge_weight, labels, attack_idx)
        # edge_index, edge_weight = convert_to_coo(A)
        adv_per_iter = self.args.adv_per_iter
        print('grad attack with {} per iter and use normalize {}'.format(adv_per_iter, self.normalize))
        num_iter = self.args.num_adv // adv_per_iter
        attack_labels = labels
        attack_labels_onehot = t.zeros((len(attack_labels), 18)) \
            .to(self.args.device) \
            .scatter_(1, attack_labels.view(-1, 1), 1)
        targeted = t.randint_like(labels, high=17).to(self.args.device)
        targeted[targeted >= labels] += 1
        num_attack = len(attack_idx)
        for it in range(num_iter):
            print('iteration {}: adding {} new node'.format(it, adv_per_iter))
            cum_adv = (it + 1) * adv_per_iter
            adv_start = X.shape[0] + it * adv_per_iter
            # adv_weight = t.ones(new).to(self.args.device)
            # opt = optim.Adam((adv_features, adv_weight), lr=1000)
            nums = [ num_attack for i in range(adv_per_iter)]
            cum = [0]
            one = []
            other = []
            for i, n in enumerate(nums):
                cum.append(cum[i] + n)
                one.extend([adv_start + i] * n)
                other.extend(attack_idx)
                # other.extend(list(range(X.shape[0], adv_start + i)))
                # other.extend(list(range(num_attack)))
                assert len(one) == len(other)
            if 'adv_features' in vars():
                cum_feat_tensor = t.cat((adv_features.data, t.randn((adv_per_iter, 100))))
            else:
                if self.args.init_feature is not None:
                    print('loading init feature')
                    init_feature = np.load(self.args.init_feature)
                    cum_feat_tensor = t.from_numpy(init_feature).float()
                else:
                    cum_feat_tensor = t.randn((adv_per_iter, 100))
            adv_features = t.autograd.Variable(cum_feat_tensor).to(self.args.device).requires_grad_(
                False)
            new_edge = t.LongTensor([one, other]).to(self.args.device)
            new_edge2 = t.LongTensor([other, one]).to(self.args.device)
            adv_weight = t.autograd.Variable(t.zeros(cum[-1])).to(self.args.device).requires_grad_(True)
            opt = optim.Adam((adv_weight,), lr=1)
            e_i = t.cat([edge_index, new_edge, new_edge2], dim=1)
            # t.nn.init.normal_(self.adv_features)
            # adv_feat = t.zeros((self.args.num_adv, 100)).to(self.args.device1)
            # features = t.autograd.Variable(t.cat([self.features, adv_feat], dim=0)).to(self.args.device1).requires_grad_(
            #     True)
            # features = normalize(features)
            for ep in range(2):
                print('start adv train on edge %d' % ep)
                if self.normalize:
                    norm_feature = 3 * F.normalize(adv_features, p=float('inf'), dim=0)
                    f = t.cat([X, norm_feature], dim=0)
                else:
                    f = t.cat([X, adv_features], dim=0)
                e_w = t.cat([edge_weight, adv_weight, adv_weight], dim=0)
                logits = self.real_model(f, e_i, e_w)
                train_logits = logits[attack_idx]
                l_val = train_logits.gather(1, attack_labels.view(-1, 1)).squeeze()
                if self.args.targeted:
                    c_val = train_logits.gather(1, targeted.view(-1, 1)).squeeze()
                else:
                    c_val, c_new = t.max(train_logits - 1e6 * attack_labels_onehot, dim=1)
                dif = c_val - l_val
                # loss = - F.cross_entropy(train_logits, attack_labels)
                loss = t.sum(dif)
                opt.zero_grad()
                # t.cuda.empty_cache()
                loss.backward()
                # --------------------- reassign weight -------------------------
                for i in range(adv_per_iter):
                    s = cum[i]
                    e = cum[i + 1]
                    w_current = adv_weight.data[s: e]
                    rel_grad = (1 - 2 * w_current) * adv_weight.grad[s: e]

                    zeros_reward = rel_grad[w_current == 0]
                    zeros_idx = t.arange(e - s)[w_current == 0]
                    zeros_s_val, zeros_s_idx = t.sort(zeros_reward, descending=True)
                    if t.sum((w_current > 0).float()) <= 0:
                        budget = 100
                        for va, id in zip(zeros_s_val, zeros_s_idx):
                            if va < 0 or budget <= 0:
                                break
                            else:
                                w_current[zeros_idx[id]] = 1
                                budget -= 1
                    else:
                        ones_reward = rel_grad[w_current > 0]
                        ones_idx = t.arange(e - s)[w_current > 0]
                        ones_s_val, ones_s_idx = t.sort(ones_reward, descending=True)
                        j = 0
                        for j, (s_id, s_va) in enumerate(zip(ones_s_idx, ones_s_val)):
                            if s_va > 0:
                                w_current[ones_idx[s_id]] = 0
                            else:
                                break

                        budget = 100 - t.sum(w_current)
                        for s_id, s_va in zip(zeros_s_idx, zeros_s_val):
                            if s_va < 0:
                                break
                            if budget <= 0:
                                if j >= len(ones_s_idx):
                                    break
                                rew = s_va + ones_s_val[j]
                                if rew <= 0:
                                    break
                                else:
                                    w_current[zeros_idx[s_id]] = 1
                                    w_current[ones_idx[ones_s_idx[j]]] = 0
                                    j += 1
                            else:
                                w_current[zeros_idx[s_id]] = 1
                                budget -= 1
                    print('total connected edge %d for node %d' % (t.sum(adv_weight.data[s: e]).item(), i))
                    assert t.sum(adv_weight.data[s: e]) <= 100
                # --------------------- reassign weight end -------------------------
                # --------------------- adv train feature ---------------------------
                adv_gt0 = adv_weight.data > 0
                non_zeros = t.sum(adv_gt0.long())
                new_one = t.LongTensor(one)[adv_gt0].tolist()
                new_other = t.LongTensor(other)[adv_gt0].tolist()
                new_edge = t.LongTensor([new_one, new_other]).to(self.args.device)
                new_edge2 = t.LongTensor([new_other, new_one]).to(self.args.device)
                tmp_edge_index = t.cat([edge_index, new_edge, new_edge2], dim=1).to(self.args.device1)
                tmp_edge_weight = t.cat([edge_weight, t.ones(2 * non_zeros)], dim=0).to(self.args.device1)
                self.real_model = self.real_model.to(self.args.device1)
                adv_features = adv_features.to(self.args.device1).requires_grad_(True)
                train_labels_gpu = attack_labels.to(self.args.device1)
                train_labels_onehot_gpu = attack_labels_onehot.to(self.args.device1)
                targeted_gpu = targeted.to(self.args.device1)
                X_gpu = X.to(self.args.device1)
                opt = optim.Adam((adv_features,), lr=0.01)
                print('start adv train on feature %d' % ep)
                for k in range(self.args.adv_feature_epoch):
                    if self.normalize:
                        norm_feature = 3 * F.normalize(adv_features, p=float('inf'), dim=0)
                        f = t.cat([X_gpu, norm_feature], dim=0)
                    else:
                        f = t.cat([X_gpu, adv_features], dim=0)
                    # e_w = t.cat([edge_weight, adv_weight, adv_weight], dim=0)
                    logits = self.real_model(f, tmp_edge_index, tmp_edge_weight)
                    train_logits = logits[attack_idx]
                    adv_acc = count_acc(train_logits, train_labels_gpu)
                    l_val = train_logits.gather(1, train_labels_gpu.view(-1, 1)).squeeze()
                    if self.args.targeted:
                        c_val = train_logits.gather(1, targeted_gpu.view(-1, 1)).squeeze()
                    else:
                        c_val, c_new = t.max(train_logits - 1e6 * train_labels_onehot_gpu, dim=1)
                    dif = l_val - c_val
                    # loss = - F.cross_entropy(train_logits, attack_labels)
                    loss = t.sum(dif)
                    opt.zero_grad()
                    # t.cuda.empty_cache()
                    loss.backward()
                    opt.step()
                    print('Epoch: {:04d} {:04d}'.format(ep + 1, k + 1),
                          'loss: {:.4f}'.format(loss.item()),
                          'acc: {:.4f}'.format(adv_acc))
                self.real_model = self.real_model.to(self.args.device)
                adv_features = t.autograd.Variable(adv_features.data).to(self.args.device).requires_grad_(
                    False)
                # --------------------- adv train feature end ---------------------------

                # adv_features.data.add_(1., adv_features.grad)
                # print('Epoch: {:04d}'.format(ep + 1),
                #       'loss: {:.4f}'.format(loss.item()),
                #       'acc: {:.4f}'.format(adv_acc))
            adv_gt0 = adv_weight.data > 0
            non_zeros = t.sum(adv_gt0.long())
            new_one = t.LongTensor(one)[adv_gt0].tolist()
            new_other = t.LongTensor(other)[adv_gt0].tolist()
            new_edge = t.LongTensor([new_one, new_other]).to(self.args.device)
            new_edge2 = t.LongTensor([new_other, new_one]).to(self.args.device)
            edge_index = t.cat([edge_index, new_edge, new_edge2], dim=1)
            edge_weight = t.cat([edge_weight, t.ones(2 * non_zeros)], dim=0)
        rows = edge_index[0, :].numpy()
        cols = edge_index[1, :].numpy()
        data = np.ones(len(rows))
        features = adv_features.data.numpy()
        total = X.shape[0] + self.args.num_adv
        adj = coo_matrix((data, (rows, cols)), shape=(total, total))
        adj = adj.tocsr()[X.shape[0]:, :]
        np.save('result/%s/feature.npy' % self.dirname, features)
        with open('result/%s/adj.pkl' % self.dirname, 'wb') as f:
            pk.dump(adj, f)

    # def normalize(self, feat):
    #     feat.data = 2 * feat.data / t.max(t.abs(feat.data), dim=0, keepdim=True)[0]

    def eval_real(self, features, edge_index, edge_weight, labels, train_idx):
        with t.no_grad():
            logits = self.real_model(features, edge_index, edge_weight)
            acc_train = count_acc(logits[train_idx], labels)
            print('real model acc: %.4f' % acc_train)
