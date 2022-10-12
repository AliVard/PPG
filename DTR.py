import numpy as np
from Mmetrics import *



class DTR():
    def __init__(self, y_pred, g, dlr, exposure, method='batch_ratio', eps = 1e-10) -> None:
        self.exposure = exposure
        self.eps = eps
        self.y_pred = y_pred
        self.g = g
        self.groups = np.unique(g)
        self.dlr = dlr
        # if dlr:
        #     self.batch_numbers = np.repeat(dlr[:-1], np.diff(dlr))
        self.eval = eval(f'self.{method}')

    def get_info(self, sorted_docs):
        return [self.y_pred[sorted_docs], self.g[sorted_docs]]

    def query_diff(self, sorted_docs, dlr):
        # print(g)
        
        qg = self.g[sorted_docs]
        qy = self.y_pred[sorted_docs]
        if len(self.g) > len(self.exposure):
            qg = qg[:len(self.exposure)]
            
        exposure = self.exposure[:len(qg)]
        qy = qy[:len(qg)]
        pred_rel = {}
        for group in self.groups:
            expo = exposure[qg==group].sum()
            pred_rel[group] = qy[qg==group].sum()
            if pred_rel[group] > self.eps:
                pred_rel[group] = expo / pred_rel[group]
            
        qdtr_pred = 0
        for i in range(len(self.groups)):
            for j in range(i+1, len(self.groups)):
                g1 = self.groups[i]
                g2 = self.groups[j]
                qdtr_pred += np.abs((pred_rel[g1]) - (pred_rel[g2]))
        return qdtr_pred
        

    def query_ratio(self, sorted_docs):
        # print(g)
        qg = self.g[sorted_docs]
        qy = self.y_pred[sorted_docs]
        if len(self.g) > len(self.exposure):
            qg = qg[:len(self.exposure)]
            
        exposure = self.exposure[:len(qg)]
        qy = qy[:len(qg)]
        pred_rel = {}
        for group in self.groups:
            expo = exposure[qg==group].sum()
            pred_rel[group] = qy[qg==group].sum()
            # print([group, expo, pred_rel[group]])
            if pred_rel[group] > self.eps:
                pred_rel[group] = expo / pred_rel[group]
            
        # print(pred_rel)
        if len(self.groups) < 2:
            return 0
        L, H = self.groups[0], self.groups[1]
        if pred_rel[L] == 0 or pred_rel[H] == 0:
            return 0
        return pred_rel[H] / pred_rel[L] if pred_rel[H] > pred_rel[L] else pred_rel[L] / pred_rel[H]

    def batch_ratio(self, sorted_docs):
        # print(dlr)
        agg_exposure = {}
        agg_utility = {}
        for group in self.groups:
            agg_exposure[group] = 0
            agg_utility[group] = 0
            
#         print('len:', len(sorted_docs))

        for qid in range(self.dlr.shape[0] - 1):
            s, e = self.dlr[qid:qid+2]

            arg = sorted_docs[s:e] - s
#             print(arg)
            qg = self.g[s:e][arg]
            qy = self.y_pred[s:e][arg]
            
            for group in self.groups:
                agg_exposure[group] += self.exposure[:len(qg)][qg==group].sum() 
                agg_utility[group] += qy[qg==group].sum() 
                # print([group, agg_exposure[group], agg_utility[group]])
                
        ratios = []
        for group in self.groups:
            if agg_utility[group] == 0:
                ratios.append(0.)
            else:
                ratios.append(agg_exposure[group] / agg_utility[group])
        if len(ratios) < 2 or ratios[0] * ratios[1] == 0:
            return 0
        DTR = ratios[0] / ratios[1] if ratios[0] > ratios[1] else ratios[1] / ratios[0]
        return DTR - 1.















    # def learn_edge_weights(epochs, lr, momentum, fn, fn_params):
    #     probs_mat = 0.5 * np.ones([len(y_pred), len(y_pred)])
    #     sorted_docs = y_pred.argsort()[::-1]
    #     sorted_g = g[sorted_docs]
    #     val = fn(sorted_docs, y_pred, g, fn_params)
    #     vals = [val]
    #     min_val = val
    #     min_val_edges = np.zeros_like(probs_mat)
    #     # print([val, sorted_docs, g[sorted_docs]])
    #     for epoch in range(epochs):
    #         docs, cnt = permute(probs_mat, sorted_g)
            
    #         # print(cnt)
    #         if cnt == 0:
    #             continue
    #         new_val = fn(sorted_docs[docs], y_pred, g, fn_params)
    #         diff = (new_val - val) / cnt
    #         if diff > 0:
    #             diff = 1. / cnt
    #         elif diff < 0:
    #             diff = -1. / cnt
    #         edges = get_edges(docs)

    #         if new_val < min_val:
    #             min_val = new_val
    #             min_val_edges = edges

    #         if False:
    #             print([new_val, sorted_docs[docs], g[sorted_docs[docs]]])
    #             print(probs_mat)
    #             print(edges)
    #         # probs_mat -= (edges) * diff * lr
    #         probs_mat -= (edges - momentum * min_val_edges) * diff * lr
    #         probs_mat[probs_mat < 0] = 0.05
    #         probs_mat[probs_mat > 1] = 0.95
    #         vals.append(new_val)
    #         val = new_val

    #     # print(min_val_edges)
    #     # print(min_val)
    #     return vals, probs_mat


def ndcg_dtr(exposure, lv, y_pred, dlr, g, query_counts):
    # print(y_pred.shape)
    groups = np.unique(g)
    # print(groups)
    agg_exposure = {}
    agg_utility = {}
    for group in groups:
        agg_exposure[group] = 0
        agg_utility[group] = 0

    for qid in range(dlr.shape[0] - 1):
        s, e = dlr[qid:qid+2]

        arg = y_pred[s:e].argsort()[::-1]
        # print(arg)
        qg = g[s:e][arg]
        qy = y_pred[s:e][arg]
        qlv = lv[s:e][arg]
        
        for group in groups:
            agg_exposure[group] += exposure[:len(qg)][qg==group].sum() * query_counts[qid]
            agg_utility[group] += qlv[qg==group].sum() * query_counts[qid]
            
    ratios = []
    for group in groups:
        ratios.append(agg_exposure[group] / agg_utility[group])
    DTR = ratios[0] / ratios[1] if ratios[0] < ratios[1] else ratios[1] / ratios[0]
    ndcg = LTRMetrics(lv,np.diff(dlr),y_pred)
    ndcg5 = ndcg.NDCG_perquery(5) * query_counts
    ndcg10 = ndcg.NDCG_perquery(10) * query_counts

    res = {'ndcg@5':ndcg5[ndcg5 > 0].sum() / query_counts[ndcg5 > 0].sum(),
            'ndcg@10':ndcg10[ndcg10 > 0].sum() / query_counts[ndcg10 > 0].sum(),
            'seq DTR':DTR,
            'single session DTR':calculatedTR(lv=lv, y_pred=y_pred, g=g, dlr=dlr)}

    return res

def calculateExposureAndUtility(lv, y_pred, g, k):

    proCount = 0
    proListX = []
    unproCount = 0
    unproListX = []
    proU = 0
    unproU = 0
    proCount = 0
    unproCount = 0
    proListX = []
    unproListX = []
    utility = []

    arg = y_pred.argsort()[::-1]
    # k = len(lv)
    # print([k, len(lv)])
    for i in range(k):
        doc = arg[i]
        if g[doc] == 'L':
            proCount += 1
            proListX.append(i)
            proU += lv[doc]
        else:
            unproCount += 1
            unproListX.append(i)
            unproU += lv[doc]

    v = np.arange(1, (k + 1), 1)
    v = 1 / np.log2(1 + v + 1)
    v = np.reshape(v, (1, k))

    v = np.transpose(v)
    proExposure = np.sum(v[proListX])
    unproExposure = np.sum(v[unproListX])

    return proExposure, unproExposure, proU, unproU, proCount, unproCount



def calculatedTR(lv, y_pred, g, dlr):
    proExposure = []
    unproExposure = []
    proUtility = []
    unproUtility = []
    proCountList = []
    unproCountList = []
    results = []
    k = 40

    for qid in range(dlr.shape[0] - 1):
        s, e = dlr[qid:qid+2]
        # if k > 40:
        #     k = 40
        #     print(
        #         'Calculation of P for k larger than 40 will not yield any results but just crash the program. Therefore k will be set to 40.')
        # if k > e-s:
        if True:
            k = e-s


        proExp, unproExp, proU, unproU, proCount, unproCount = calculateExposureAndUtility(lv = lv[s:e], y_pred = y_pred[s:e], g = g[s:e], k=k)
        proExposure.append(proExp)
        unproExposure.append(unproExp)
        proUtility.append(proU)
        unproUtility.append(unproU)
        proCountList.append(proCount)
        unproCountList.append(unproCount)

    top = 0
    bottom = 0

    # calculate value for each group
    if sum(proCountList) != 0:
        proU = sum(proUtility) / sum(proCountList)
        proExposure = sum(proExposure) / sum(proCountList)
        top = (proExposure / proU)

    if sum(unproCountList) != 0:
        unproU = sum(unproUtility) / sum(unproCountList)
        unproExposure = sum(unproExposure) / sum(unproCountList)
        bottom = (unproExposure / unproU)

    # calculate DTR
    dTR_origin = top / bottom
    return dTR_origin if dTR_origin < 1 else 1./dTR_origin
