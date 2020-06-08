import torch
import torch.nn.functional as F
import time
import torch.optim as optim

from pygcn.utils import accuracy


class GcnTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, data, labels, idx_train, idx_val, epoch_num):
        for epoch in range(epoch_num):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            loss_train.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
                acc_val = accuracy(output[idx_val], labels[idx_val])

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def test(self, data, labels, idx_test):
        self.model.eval()
        output = self.model(data)
        loss_test = F.cross_entropy(data[idx_test], labels[idx_test])
        acc_test = accuracy(output, labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
