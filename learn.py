
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from collections import defaultdict
import sys

from torch.utils.data import Dataset, DataLoader
from Mmetrics import *

import LTR
import datautil
import permutationgraph
import DTR
import EEL
import PPG
import PL

ds2020, _ = datautil.load_data(2020, verbose=False)
ds2019, _ = datautil.load_data(2019, verbose=False)

ltrmodel = LTR.MSE_model(layers=[ds2020.trfm.shape[1], 256, 256, 1], lr=0.001, optimizer=torch.optim.Adam, dropout=0.1)
ltrmodel.fit(ds2020, epochs=10, batch_size=100, verbose=False)
y_pred2020 = ltrmodel.predict(ds2020.tefm, ds2020.tedlr)
print('LTR performance ndcg@10 for 2020:', LTRMetrics(ds2020.telv,np.diff(ds2020.tedlr),y_pred2020).NDCG(10))

ltrmodel = LTR.MSE_model(layers=[ds2019.trfm.shape[1], 256, 256, 1], lr=0.001, optimizer=torch.optim.Adam, dropout=0.1)
ltrmodel.fit(ds2019, epochs=10, batch_size=100, verbose=False)
y_pred2019 = ltrmodel.predict(ds2019.tefm, ds2019.tedlr)
print('LTR performance ndcg@10 for 2019:', LTRMetrics(ds2019.telv,np.diff(ds2019.tedlr),y_pred2019).NDCG(10))

epochs = 50


alg = 'PPG'
if len(sys.argv) > 1:
    alg = sys.argv[1]

metric = 'EEL'
if len(sys.argv) > 2:
    metric = sys.argv[2]
    

intra = False
suffix = 'nointra_'
if len(sys.argv) > 3:
    if sys.argv[3] == 'intra':
        intra = True
        suffix='intra_'
        
        
samples_cnt=32
if len(sys.argv) > 4:
    samples_cnt = eval(sys.argv[4])

    
sessions_cnt=20
if len(sys.argv) > 5:
    sessions_cnt = eval(sys.argv[5])

        
suffix += f'{alg}_{samples_cnt}_{sessions_cnt}'



exposure2020 = np.array([1./np.log2(2+i) for i in range(1,np.diff(ds2020.tedlr).max()+2)])
exposure2019 = np.array([1./np.log2(2+i) for i in range(1,np.diff(ds2019.tedlr).max()+2)])


def learn_one_PPG(metric, qid, verbose, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    s, e = dlr[qid:qid+2]
    y_pred_s, g_s, sorted_docs_s, dlr_s = \
        EEL.copy_sessions(y=y_pred[s:e], g=g[s:e], sorted_docs=y_pred[s:e].argsort()[::-1], sessions=sessions_cnt)
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = y_pred_s, g = g_s, dlr = dlr_s, exposure=exposure, grade_levels = grade_levels)
    else:
        objective_ins = DTR.DTR(y_pred = y_pred_s, g = g_s, dlr = dlr_s, exposure=exposure)
        
    learner = PPG.Learner(  PPG_mat=None, samples_cnt=samples_cnt, 
                                objective_ins=objective_ins, 
                                sorted_docs = sorted_docs_s, 
                                dlr = dlr_s,
                                intra = g_s if intra else np.arange(g_s.shape[0]),
                                inter = np.repeat(dlr_s[:-1], np.diff(dlr_s)))
    vals = learner.fit(epochs, lr, verbose=verbose)
    return vals

def learn_all_PPG(metric, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    sorted_docs = []
    
#     for qid in trange(dlr.shape[0] - 1, leave=False):
    for qid in range(dlr.shape[0] - 1):
        min_b = learn_one_PPG(metric, qid, 0, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt)
        sorted_docs.append(min_b)
        

    # print(ndcg_dtr(exposure, lv, np.concatenate(y_rerank), dlr, g, query_counts))
    return sorted_docs


def learn_one_PL(metric, qid, verbose, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    s, e = dlr[qid:qid+2]
    
    if metric == 'EEL':
        objective_ins = EEL.EEL(y_pred = y_pred[s:e], g = g[s:e], dlr = np.array([0,e-s]), exposure=exposure, grade_levels = grade_levels)
    else:
        objective_ins = DTR.DTR(y_pred = y_pred[s:e], g = g[s:e], dlr = np.array([0,e-s]), exposure=exposure)
        
    learner = PL.Learner(logits=y_pred[s:e], samples_cnt=samples_cnt, 
                        objective_ins=objective_ins, sessions_cnt=sessions_cnt)
    vals = learner.fit(epochs, lr, verbose=verbose)
    return vals

def learn_all_PL(metric, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt):
    sorted_docs = []
    
#     for qid in trange(dlr.shape[0] - 1, leave=False):
    for qid in range(dlr.shape[0] - 1):
        min_b = learn_one_PL(metric, qid, 0, y_pred, g, dlr, epochs, lr, exposure, grade_levels, samples_cnt, sessions_cnt)
        sorted_docs.append(min_b)
        

    # print(ndcg_dtr(exposure, lv, np.concatenate(y_rerank), dlr, g, query_counts))
    return sorted_docs


learn_fn = eval(f'learn_all_{alg}')

res = {}

for learning_rate in ['0.01', '0.1', '0.5']:
    res[f'2020_{learning_rate}'] = \
        learn_fn(metric, y_pred2020, ds2020.teg, ds2020.tedlr, epochs, eval(learning_rate), exposure=exposure2020,
        grade_levels=5, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)

    with open(f'/ivi/ilps/personal/avardas/_data/PPG/{suffix}_{metric}_results.pkl', 'wb') as f:
        pickle.dump(res, f)

    res[f'2019_{learning_rate}'] = \
        learn_fn(metric, y_pred2019, ds2019.teg, ds2019.tedlr, epochs, eval(learning_rate), exposure=exposure2019,
        grade_levels=5, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)
    
    with open(f'/ivi/ilps/personal/avardas/_data/PPG/{suffix}_{metric}_results.pkl', 'wb') as f:
        pickle.dump(res, f)
    
#     res[f'lv_2020_{learning_rate}'] = \
#         learn_fn(ds2020.telv, ds2020.teg, ds2020.tedlr, epochs, eval(learning_rate), exposure=exposure2020,
#         grade_levels=2, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)

#     with open(f'{suffix}_EEL_results.pkl', 'wb') as f:
#         pickle.dump(res, f)

#     res[f'lv_2019_{learning_rate}'] = \
#         learn_fn(ds2019.telv, ds2019.teg, ds2019.tedlr, epochs, eval(learning_rate), exposure=exposure2019,
#         grade_levels=2, samples_cnt=samples_cnt, sessions_cnt=sessions_cnt)
    
    
#     with open(f'{suffix}_EEL_results.pkl', 'wb') as f:
#         pickle.dump(res, f)