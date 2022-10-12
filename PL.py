# based on Sean Robertson's pydrobert-pytorch.estimators


import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import logcumsumexp, reverse_logcumsumexp, smart_perm, make_permutation_matrix
import torch
from torch.distributions.distribution import Distribution
from torch.distributions import constraints

import EEL

import numpy as np

def to_z(logits, u=None):
    if u is not None:
        assert u.size() == logits.size()
    else:
        u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
    log_probs = F.log_softmax(logits, dim=-1)
    z = log_probs - torch.log(-torch.log(u))
    z.requires_grad_(True)
    return z

def to_b(z):
    b = torch.sort(z, descending=True, dim=-1)[1]
    return b


class PlackettLuce(Distribution):
    """
        Plackett-Luce distribution
    """
    arg_constraints = {"logits": constraints.real}

    def __init__(self, logits):
        # last dimension is for scores of plackett luce
        super(PlackettLuce, self).__init__()
        self.logits = logits
        self.size = self.logits.size()

    def sample(self, num_samples):
        # sample permutations using Gumbel-max trick to avoid cycles
        with torch.no_grad():
            logits = self.logits.unsqueeze(0).expand(num_samples, *self.size)
            u = torch.distributions.utils.clamp_probs(torch.rand_like(logits))
            z = self.logits - torch.log(-torch.log(u))
            samples = torch.sort(z, descending=True, dim=-1)[1]
        return samples

    def log_prob(self, samples):
        # samples shape is: num_samples x self.size
        # samples is permutations not permutation matrices
        if samples.ndimension() == self.logits.ndimension():  # then we already expanded logits
            logits = smart_perm(self.logits, samples)
        elif samples.ndimension() > self.logits.ndimension():  # then we need to expand it here
            logits = self.logits.unsqueeze(0).expand(*samples.size())
            logits = smart_perm(logits, samples)
        else:
            raise ValueError("Something wrong with dimensions")
        logp = (logits - reverse_logcumsumexp(logits, dim=-1)).sum(-1)
        return logp



def reinforce(fb, b, logits, **kwargs):
    b = b.detach()
    log_pb = PlackettLuce(logits=logits).log_prob(b)
    g = fb * torch.autograd.grad([log_pb], [logits], grad_outputs=torch.ones_like(log_pb))[0]
    g = g.unsqueeze(0)
    return g


class Learner:
    def __init__(self, logits, samples_cnt, objective_ins, sessions_cnt) -> None:
        self.log_theta = torch.tensor(EEL.linspan(logits, 5), requires_grad=True)
        self.objective = objective_ins
        self.n = len(logits)
        self.samples_cnt = samples_cnt
        self.sessions_cnt = sessions_cnt
        

    def fit(self, epochs, lr, verbose):
        optim = torch.optim.Adam([self.log_theta], lr)

        for i in range(epochs):
            optim.zero_grad()
            d_log_thetas = []
            fbs = 0
            for _ in range(self.samples_cnt):
                u = torch.distributions.utils.clamp_probs(torch.rand_like(self.log_theta))
                z = to_z(self.log_theta, u)
                b = to_b(z)
                f_b = self.objective.eval(b.detach().data.numpy())
                if verbose > 1:
                    print(b.detach().data.numpy(), f_b)
                fbs += f_b / self.samples_cnt
                d_log_thetas.append(reinforce(fb=f_b, b=b, logits=self.log_theta))
            d_log_thetas = torch.cat(d_log_thetas,0)
            if verbose > 0:
                print(fbs)
            self.log_theta.backward(d_log_thetas.mean(0))
            optim.step()

        output = []
        for i in range(self.sessions_cnt):
            u = torch.distributions.utils.clamp_probs(torch.rand_like(self.log_theta))
            z = to_z(self.log_theta, u)
            b = to_b(z) + (i * self.n)
            output.append(b)
        return np.concatenate(output)
            