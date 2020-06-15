import torch
import torch.nn.functional as F
import time
import torch.optim as optim

from pygcn.utils import accuracy
from torch_geometric.data import DataLoader


class GcnTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

    def train(self, data, epoch_num):
        my_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=100, gamma=0.9)
        for epoch in range(epoch_num):
            t = time.time()
            self.model.train()
            self.optimizer.zero_grad()
            output = self.model(data)
            loss_train = F.cross_entropy(output[data.train_mask], data.y[data.train_mask])
            acc_train = accuracy(output[data.train_mask], data.y[data.train_mask])
            loss_train.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                loss_val = F.cross_entropy(output[data.val_mask], data.y[data.val_mask])
                acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])

            my_lr_scheduler.step()
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def train_cluster(self, cluster_data, epoch_num):
        loader = DataLoader(cluster_data, shuffle=True, batch_size=1)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.9)

        for epoch in range(epoch_num):
            t = time.time()
            for batch, data in enumerate(loader):
                self.model.train()
                self.optimizer.zero_grad()
                output = self.model(cluster_data.data)
                loss_train = F.cross_entropy(output[cluster_data.train_mask], cluster_data.y[cluster_data.train_mask])
                acc_train = accuracy(output[cluster_data.train_mask], cluster_data.y[cluster_data.train_mask])
                loss_train.backward()
                self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                loss_val = F.cross_entropy(output[cluster_data.val_mask], cluster_data.y[cluster_data.val_mask])
                acc_val = accuracy(output[cluster_data.val_mask], cluster_data.y[cluster_data.val_mask])

            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def test(self, data):
        self.model.eval()
        output = self.model(data)
        loss_test = F.cross_entropy(output[data.test_mask], data.y[data.test_mask])
        acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        torch.save(self.model, "./SAGE_nonorm_{:.5f}.pkl".format(acc_test))
