import time
import lightgbm as lgb
import Mmetrics
import os
import numpy as np


def read_pkl(pkl_path):
    loaded_data = np.load(pkl_path, allow_pickle=True)
    feature_map = loaded_data['feature_map'].item()
    train_feature_matrix = loaded_data['train_feature_matrix']
    train_doclist_ranges = loaded_data['train_doclist_ranges']
    train_label_vector   = loaded_data['train_label_vector']
    valid_feature_matrix = loaded_data['valid_feature_matrix']
    valid_doclist_ranges = loaded_data['valid_doclist_ranges']
    valid_label_vector   = loaded_data['valid_label_vector']
    test_feature_matrix  = loaded_data['test_feature_matrix']
    test_doclist_ranges  = loaded_data['test_doclist_ranges']
    test_label_vector    = loaded_data['test_label_vector']
    dataset = type('', (), {})()
    dataset.fmap = feature_map
    dataset.trfm = train_feature_matrix
    dataset.tefm = test_feature_matrix
    dataset.vafm = valid_feature_matrix
    dataset.trdlr = train_doclist_ranges
    dataset.tedlr = test_doclist_ranges
    dataset.vadlr = valid_doclist_ranges
    dataset.trlv = train_label_vector
    dataset.telv = test_label_vector
    dataset.valv = valid_label_vector
    
    print('num features : {}'.format(dataset.trfm.shape[1]))
    print('num docs (train, valid, test) : ({},{},{})'.format(dataset.trfm.shape[0], dataset.vafm.shape[0], dataset.tefm.shape[0]))
    print('num queries (train, valid, test) : ({},{},{})'.format(dataset.trdlr.shape[0], dataset.vadlr.shape[0], dataset.tedlr.shape[0]))

    return dataset
    

def lambdarank(dataset, model_path=None, learning_rate=0.05, num_leaves=31, n_estimators=300, eval_at=[10], early_stopping_rounds=10000):
    start = time.time()
    if model_path is not None and os.path.exists(model_path):
        booster = lgb.Booster(model_file=model_path)
        print('loading lgb took {} secs.'.format(time.time() - start))
        return booster

    gbm = lgb.LGBMRanker(learning_rate=learning_rate, n_estimators=n_estimators, num_leaves=num_leaves)

    gbm.fit(dataset.trfm, dataset.trlv, 
          group=np.diff(dataset.trdlr), 
          eval_set=[(dataset.vafm, dataset.valv)],
          eval_group=[np.diff(dataset.vadlr)], 
          eval_at=eval_at, 
          early_stopping_rounds=early_stopping_rounds, 
          verbose=False)

    if model_path is not None:
        gbm.booster_.save_model(model_path)

    print('training lgb took {} secs.'.format(time.time() - start))
    return gbm.booster_

def predict(booster, dataset):
#     start = time.time()
    y_pred = booster.predict(dataset.tefm)
#     print('predict took {} secs.'.format(time.time() - start))
    return y_pred

def evaluate(dataset, booster):
    y_pred = predict(booster, dataset)
    metric = metrics.LTRMetrics(dataset.telv,np.diff(dataset.tedlr),y_pred)

    return metric.NDCG(10)

def evaluate_valid(dataset, booster):
    y_pred = booster.predict(dataset.vafm)
    metric = metrics.LTRMetrics(dataset.valv,np.diff(dataset.vadlr),y_pred)

    return metric.NDCG(10)

def evaluate_train(dataset, booster):
    y_pred = booster.predict(dataset.trfm)
    metric = metrics.LTRMetrics(dataset.trlv,np.diff(dataset.trdlr),y_pred)

    return metric.NDCG(10)
