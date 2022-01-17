import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from Mmetrics import *

class MLTRData(Dataset):
    def __init__(self, fm, lv):
        self.feature = fm
        self.labels = lv
        self.dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if torch.cuda.is_available():
            self.torch_ = torch.cuda
        else:
            self.torch_ = torch
            
    def _get_tensor(self, arr):
        return self.torch_.FloatTensor(arr, device=self.dev)

    def __len__(self):
        return self.labels.shape[0] - 1
    def __getitem__(self, i):
        feature = self._get_tensor(self.feature[i,:])
        labels = self._get_tensor(self.labels[i:i+1])
        return feature, labels
    

class DNN(nn.Module):
    def __init__(self, layers_size, dropout):
        super(DNN, self).__init__()
        layers = []
        for i in range(len(layers_size)-2):
            layers.append(nn.Linear(layers_size[i], layers_size[i+1], bias=True))
            if dropout is not None:
                layers.append(nn.Dropout(p=dropout, inplace=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(layers_size[-2], layers_size[-1], bias=False))
        self.layers = nn.Sequential(*layers)
        if torch.cuda.is_available():
            self.layers.cuda()
    
    def forward(self, x):
        return self.layers(x)

    def train_batch(self, x, y, optimizer, loss_fn):
        self.train()
        optimizer.zero_grad()
        # print(x.shape)
        # print(y.shape)
        out = self.forward(x)#[:,0]
        # print(out.shape)
        loss = loss_fn(out, y)
        loss.backward()

        optimizer.step()

        return loss

class MSE_model():
    def __init__(self, layers, optimizer, lr, dropout):
        self.net = DNN(layers, dropout)
        self.opt = optimizer(self.net.parameters(), lr=lr)

    def fit(self, dataset, epochs, batch_size, shuffle=True, verbose=True):
        train_data = MLTRData(dataset.trfm, dataset.trlv)
        dl = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        loss_fn_ = torch.nn.MSELoss()
        loss_fn = lambda out, y: loss_fn_(out, torch.sigmoid(y))

        for epoch in range(epochs):
            loss = []
            for (x,y) in dl:
                batch_loss = self.net.train_batch(x, y, self.opt, loss_fn)
                loss.append(batch_loss.data.cpu()) 

            if verbose:
                self.net.eval()
                y_pred = self.net(train_data._get_tensor(dataset.tefm)).data.cpu().numpy()[:,0]
                ndcg = LTRMetrics(dataset.telv,np.diff(dataset.tedlr),y_pred)
                y_trpred = self.net(train_data._get_tensor(dataset.trfm)).data.cpu().numpy()[:,0]
                trndcg = LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),y_pred)
                print(f'epoch {epoch} -> loss: {np.array(loss).mean()}, train ndcg@10: {trndcg.NDCG(10)}, ndcg@10: {ndcg.NDCG(10)}')
        
    def predict(self, fm, dlr):
        y = self.net(MLTRData(None, None)._get_tensor(fm))
        y = torch.sigmoid(y)
        return y.data.cpu().numpy()[:,0]
