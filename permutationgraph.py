import numpy as np


def get_edges(args):
    edges = np.zeros([len(args), len(args)])
    for i in range(len(args)):
        for j in range(i+1, len(args)):
            if args[i] > args[j]:
                edges[args[j]][args[i]] = 1
    return edges


class QueryLearner():      
    def __init__(self, objective_ins, sorted_docs, intra) -> None:
        self.sorted_docs = sorted_docs
        self.g = intra
        self.objective = objective_ins
        self.n = len(self.sorted_docs)

    def _swap_if(self, docs, i, visited, changed):
        # print([i, docs, docs[i:i+2]])
        d1, d2 = sorted(docs[i:i+2])
        if self.g[self.sorted_docs[d1]] != self.g[self.sorted_docs[d2]] and visited[d1][d2] == 0:
            if self.verbose:
                print([i, d1, d2, self.probs_mat[d1][d2], visited[d1][d2]])
            visited[d1][d2] = 1
            if np.random.binomial(1, self.probs_mat[d1][d2]):
                if self.verbose:
                    print('x')
                tmp = docs[i]
                docs[i] = docs[i+1]
                docs[i+1] = tmp
                changed.append(i)
                return True
        return False

    def _permute(self):
        docs = np.arange(self.n)
        visited = np.zeros_like(self.probs_mat)
        if self.verbose:
            print('-------------')
        changed = list(range(self.n))
        inversion_cnt = 0
        
        while changed:
            candid_set = set()
            for i in changed:
                if i - 1 >= 0:
                    candid_set.add(i-1)
                if i + 1 < len(docs) - 1:
                    candid_set.add(i+1)
            # print(candid_set)
            candid_set = np.random.permutation(list(candid_set))
            # print(candid_set)
            changed = []
            skip = set()
            for i in candid_set:
                if i in skip:
                    continue
                if self._swap_if(docs, i, visited, changed):
                    skip.add(i+1)
                    skip.add(i-1)
                    # pass
                    
            inversion_cnt += len(changed)
            # break
        return docs, inversion_cnt

        
    def fit(self, epochs, lr, verbose=False):
        self.verbose = verbose

        val = self.objective.eval(self.sorted_docs)
        vals = [val]
        reordered = True
        if verbose:
            print([val, self.sorted_docs])
        min_val = val
        min_val_docs = self.sorted_docs
        vals_list = [val]
        for epoch in range(epochs):
            if reordered:
                self.probs_mat = (0.5**(1./2.)) * np.ones([self.n, self.n])
                reordered = False
            docs, cnt = self._permute()
            
            # print(cnt)
            if cnt == 0:
                continue
            new_val = self.objective.eval(self.sorted_docs[docs])
            vals_list.append(new_val)

            if new_val < min_val:
                min_val = new_val
                min_val_docs = self.sorted_docs[docs]
            # if new_val < min_val:
            if new_val < val:
                # self.sorted_docs = self.sorted_docs[docs]
                # reordered = True
                # min_val = new_val
                diff = max((new_val - val) / np.array(vals_list).mean(), -1.)
                edges = get_edges(docs)
                self.probs_mat -= (edges) * diff * lr
                self.probs_mat[self.probs_mat < 0] = 0.01
                self.probs_mat[self.probs_mat > 1] = 0.99
            # elif new_val > min_val:
            elif new_val > val:
                # diff = min((new_val - min_val) / np.array(vals_list).mean(), 1.)
                diff = min((new_val - val) / np.array(vals_list).mean(), 1.)
                edges = get_edges(docs)
                self.probs_mat -= (edges) * diff * lr
                self.probs_mat[self.probs_mat < 0] = 0.01
                self.probs_mat[self.probs_mat > 1] = 0.99

            if verbose:
                if min_val == new_val:
                    print([new_val, self.sorted_docs] + self.objective.get_info(self.sorted_docs))
                else:
                    print([min_val, '<', new_val, self.sorted_docs[docs]] + self.objective.get_info(self.sorted_docs[docs]))
                    print(self.probs_mat)
                # print(edges)
            # probs_mat -= (edges) * diff * lr
            
            vals.append(new_val)
            val = new_val
        self.sorted_docs = min_val_docs
        return vals






class BatchLearner():
    def __init__(self, objective_ins, sorted_docs, intra, inter) -> None:
        self.sorted_docs = sorted_docs
        
        self.g = intra
        self.batch_numbers = inter
        self.objective = objective_ins
        self.n = len(sorted_docs)

    def _swap_if(self, docs, i, visited, changed):
        # print([i, docs, docs[i:i+2]])
        d1, d2 = sorted(docs[i:i+2])
        if self.g[self.batch_numbers[d1] + self.sorted_docs[d1]] != self.g[self.batch_numbers[d2] + self.sorted_docs[d2]] and \
                self.batch_numbers[d1] == self.batch_numbers[d2] and \
                    visited[d1][d2] == 0:
            visited[d1][d2] = 1
            if np.random.binomial(1, self.probs_mat[d1][d2]):
                tmp = docs[i]
                docs[i] = docs[i+1]
                docs[i+1] = tmp
                changed.append(i)
                return True
        return False
    

    def _permute(self):
        docs = np.arange(self.n)
        visited = np.zeros_like(self.probs_mat)
        changed = list(range(self.n))
        inversion_cnt = 0
        
        while changed:
            candid_set = set()
            for i in changed:
                if i - 1 >= 0:
                    candid_set.add(i-1)
                if i + 1 < len(docs) - 1:
                    candid_set.add(i+1)
            # print(candid_set)
            candid_set = np.random.permutation(list(candid_set))
            # print(candid_set)
            changed = []
            skip = set()
            for i in candid_set:
                if i in skip:
                    continue
                if self._swap_if(docs, i, visited, changed):
                    skip.add(i+1)
                    skip.add(i-1)
                    # pass
                    
            inversion_cnt += len(changed)
        return docs, inversion_cnt


    def fit(self, epochs, lr, verbose=False):
        val = self.objective.eval(self.sorted_docs)
        
        vals = [val]
        reordered = True
        if verbose:
            print(self.sorted_docs)
            print(self.batch_numbers)
            print([val, self.sorted_docs])
        min_val = val
        for epoch in range(epochs):
            if reordered:
                self.probs_mat = 0.5 * np.ones([self.n, self.n])
                reordered = False
            docs, cnt = self._permute()
            
            # print(cnt)
            if cnt == 0:
                continue
            new_val = self.objective.eval(self.sorted_docs[docs])

            if new_val < min_val:
                self.sorted_docs = self.sorted_docs[docs]
                reordered = True
                min_val = new_val
            else:
                diff = new_val - min_val
                edges = get_edges(docs)
                self.probs_mat -= (edges) * diff * lr
                self.probs_mat[self.probs_mat < 0] = 0.01
                self.probs_mat[self.probs_mat > 1] = 0.99

            if verbose:
                if min_val == new_val:
                    print([new_val, self.sorted_docs] + self.objective.get_info(self.sorted_docs))
                else:
                    print([min_val, '<', new_val, self.sorted_docs[docs]] + self.objective.get_info(self.sorted_docs[docs]))
                    # print(probs_mat)
                # print(edges)
            
            vals.append(new_val)
            val = new_val
        return vals


def get_group_counts(g, dlr):
    groups = np.unique(g)
    gcnt = [[] for _ in range(len(groups))]
    for qid in range(dlr.shape[0] - 1):
        s, e = dlr[qid:qid+2]
        for i, group in enumerate(groups):
            gcnt[i].append(len(np.where(g[s:e] == group)[0]))
    for i, group in enumerate(groups):
            gcnt[i] = np.array(gcnt[i])
    return groups, gcnt
