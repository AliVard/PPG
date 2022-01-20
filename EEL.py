import numpy as np


def linspan(y_pred, levels):
    m,M = y_pred.min()-1e-10, y_pred.max()+1e-10
    m = max(m,0)
    step = (M - m) / levels
    return np.floor((y_pred-m)/step)

def disc_target_exposure(y, exposure):
    sorted_y = np.sort(y)[::-1]
    expo = exposure[:len(y)]
    te = []
    for g in range(int(y.max()+1)):
        if len(expo[sorted_y==g]) == 0:
            te.append(0)
        else:
            te.append(np.mean(expo[sorted_y==g]))
    return np.array(te)

def copy_sessions(y, g, sorted_docs, sessions):
    ys, gs, sorteds, dlrs = [], [], [], [0]
    for i in range(sessions):
        ys.append(y.copy())
        gs.append(g.copy())
        sorteds.append((i*y.shape[0]) + sorted_docs.copy())
        dlrs.append((i+1)*y.shape[0])
    return np.concatenate(ys), np.concatenate(gs), np.concatenate(sorteds), np.array(dlrs)

class EEL:
    def __init__(self, y_pred, g, dlr, exposure, grade_levels, eps = 1e-10) -> None:
        self.exposure = exposure
        self.eps = eps
        self.y_pred = y_pred
        self.g = g
        self.grade_levels = grade_levels
        self.groups = np.unique(g)
        self.dlr = dlr
        self.target_exp = self._compute_target_exposure()
#         print(self.g)
#         print(self.target_exp)

    def get_info(self, sorted_docs):
        return [self.y_pred[sorted_docs], self.g[sorted_docs]]
        
    def _compute_target_exposure(self):
        agg_exposure = {}
        for group in self.groups:
            agg_exposure[group] = 0

        for qid in range(self.dlr.shape[0] - 1):
            s, e = self.dlr[qid:qid+2]
#             print('target', s, e)
            qg = self.g[s:e]
            disc_y = linspan(self.y_pred[s:e], self.grade_levels)
            level_target_exposure = disc_target_exposure(disc_y, self.exposure)
            item_exposure = level_target_exposure[disc_y.astype(int)]
            for group in self.groups:
                if len(item_exposure[qg==group]) == 0:
                    agg_exposure[group] += 0
                else:
                    agg_exposure[group] += item_exposure[qg==group].sum() / (self.dlr.shape[0] - 1)
        return agg_exposure

    def _expected_exposure(self, sorted_docs):
        agg_exposure = {}
        for group in self.groups:
            agg_exposure[group] = 0

        for qid in range(self.dlr.shape[0] - 1):
            s, e = self.dlr[qid:qid+2]
#             print('expected', s, e)
            arg = sorted_docs[s:e] - s
            # print(arg)
            qg = self.g[s:e][arg]
            
            for group in self.groups:
                if len(qg[qg==group]) == 0:
                    agg_exposure[group] += 0
                else:
                    agg_exposure[group] += self.exposure[:len(qg)][qg==group].sum() / (self.dlr.shape[0] - 1)
                # print([group, agg_exposure[group], agg_utility[group]])
        return agg_exposure   

    def eval(self, sorted_docs):
        exp = self._expected_exposure(sorted_docs)
        delta = []
        for group in self.groups:
            delta.append(exp[group] - self.target_exp[group])
#         print(delta)
        return np.square(np.array(delta)).sum()

    def eval_detailed(self, sorted_docs):
        exp = self._expected_exposure(sorted_docs)
        tr, ex = [], []
        for group in self.groups:
            ex.append(exp[group])
            tr.append(self.target_exp[group])
        ex = np.array(ex)
        tr = np.array(tr)
        
        return np.square(ex-tr).sum(), np.square(ex).sum(), np.dot(ex,tr).sum()