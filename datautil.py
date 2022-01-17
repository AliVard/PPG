import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
from Mmetrics import *

# from ListNet.runListNet import runListNet
# from candidateCreator.createCandidate import createCandidate as cC

def readcsv(path):
    fm, lv, dlr, qids, g = [], [], [], [], []
    qid = -1
    with open(path, 'r') as f:
        f.readline()
        i = 0
        for line in f:
            cols = line.split(',')
            g.append(cols[0])
            n_qid = cols[1]
            if n_qid != qid:
                dlr.append(i)
                qids.append(n_qid)
                qid = n_qid
            lv.append(float(cols[2]))
            fm.append(np.array(list(map(float, cols[3:])))[None, :])
            i += 1
    dlr.append(i)
    return np.concatenate(fm, 0), np.array(lv), np.array(dlr), np.array(g), np.array(qids)

def readseq(path):
    counts = defaultdict(lambda : 0)
    with open(path, 'r') as f:
        for line in f:
            counts[line.split(',')[1][:-1]] += 1
    return counts
def normalize(fm, dlr):
    for qid in range(dlr.shape[0] - 1):
        s, e = dlr[qid:qid+2]
        m = fm[s:e,:].mean(axis=0)
        z = fm[s:e,:].std(axis=0) + 1e-10
        fm[s:e, :] = (fm[s:e, :] - m) / z

def normalize2(fm, dlr):
    for qid in range(dlr.shape[0] - 1):
        s, e = dlr[qid:qid+2]
        m = fm[s:e,:].min(axis=0)
        z = fm[s:e,:].max(axis=0) - m + 1e-10
        fm[s:e, :] = (fm[s:e, :] - m) / z
    


def load_data(year=2020, normalization='Gaussian', verbose=True):
    dataset = {}

    dataset['trfm'], dataset['trlv'], dataset['trdlr'], dataset['trg'], dataset['trqid'] = readcsv(f'{year}/train.csv')
    dataset['tefm'], dataset['telv'], dataset['tedlr'], dataset['teg'], dataset['teqid'] = readcsv(f'{year}/test.csv')

    query_counts_ids = readseq(f'{year}/query_seq_10000.csv')
    query_counts = np.zeros(dataset['tedlr'].shape[0] - 1)
    for qid in range(dataset['tedlr'].shape[0] - 1):
        query_counts[qid] = query_counts_ids[dataset['teqid'][qid]]

    dataset['query_seq'] = query_counts
    ds = type('ltr', (object,), dataset)

    if verbose:
        print(f"train: {dataset['trfm'].shape[0]} docs, {dataset['trdlr'].shape[0] - 1} queries.")
        print(f"test: {dataset['tefm'].shape[0]} docs, {dataset['tedlr'].shape[0] - 1} queries.")
        print(f"{dataset['trfm'].shape[1]} features")
        print(f'un-normalized train: {ds.trfm.mean(axis=0).mean()} <{ds.trfm.std(axis=0).mean()}>')
        print(f'un-normalized test: {ds.tefm.mean(axis=0).mean()} <{ds.tefm.std(axis=0).mean()}>')

    if normalization == 'Gaussian':
        normalize(ds.trfm, ds.trdlr)
        normalize(ds.tefm, ds.tedlr)
    elif normalization == 'minmax':
        normalize2(ds.trfm, ds.trdlr)
        normalize2(ds.tefm, ds.tedlr)
    elif normalization is not None:
        print(f'{normalization} normalizaton not recongnized!')

    if normalization and verbose:
        print(f'normalized train: {ds.trfm.mean(axis=0).mean()} <{ds.trfm.std(axis=0).mean()}>')
        print(f'normalized test: {ds.tefm.mean(axis=0).mean()} <{ds.tefm.std(axis=0).mean()}>')

    return ds, dataset




def compute_dtr(exposure, lv, y_pred, dlr, g, eps = 1e-10):
    groups = np.unique(g)
    dtr_pred, dtr_true = 0, 0
    for qid in range(dlr.shape[0] - 1):
        s, e = dlr[qid:qid+2]
        arg = y_pred[s:e].argsort()[::-1]
        qg = g[s:e][arg]
        qy = y_pred[s:e][arg]
        qlv = lv[s:e][arg]
        if e - s > len(exposure):
            qg = qg[:len(exposure)]
            
            
        pred_rel, true_rel = {}, {}
        for group in groups:
            expo = exposure[:len(qg)][qg==group].sum()
            pred_rel[group] = qy[:len(qg)][qg==group].sum()
            if pred_rel[group] > eps:
                pred_rel[group] = expo / pred_rel[group]
            true_rel[group] = qlv[:len(qg)][qg==group].sum()
            if true_rel[group] > eps:
                true_rel[group] = expo / true_rel[group]
        qdtr_pred, qdtr_true = 0, 0
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                g1 = groups[i]
                g2 = groups[j]
                qdtr_pred += np.abs((pred_rel[g1]) - (pred_rel[g2]))
                qdtr_true += np.abs((true_rel[g1]) - (true_rel[g2]))
        dtr_pred += qdtr_pred
        dtr_true += qdtr_true
    return dtr_pred/(dlr.shape[0] - 1), dtr_true/(dlr.shape[0] - 1)
        
