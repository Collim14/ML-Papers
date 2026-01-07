import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

class SinkhornUtil:
    @staticmethod
    def apply(login, iter = 20, eps = 1e-6):
        M = torch.exp(login)
        for _ in range(iter):
            M = M / (M.sum(dim = -1, keepdim = True)+eps)
            M= M / (M.sum(dim = -2, keepdim = True)+eps)
        return M

    

    @staticmethod
    def applylog(login, iter = 20, eps = 1e-6):
        logup = torch.zeros((login.shape[-1],), device=login.device, dtype=login.dtype)
        r = torch.zeros((login.shape[:-1],), device=login.device, dtype=login.dtype)
        c = torch.zeros((login.shape[:-1],), device=login.device, dtype=login.dtype)
        for _ in range(iter):
            r = logup - torch.logsumexp(login + r.unsqueeze(-2), dim=-1)
            c = logup - torch.logsumexp(login + c.unsqueeze(-1), dim=-2)

        return torch.exp(login +r.unsqueeze(-1)+c.unsqueeze(-2))


class mHCModule(nn.Module):
    def __init__(self, n_streams, module, device=None):
        super().__init__()
        self.n_streams = n_streams
        self.F = module
        self.device = device
        