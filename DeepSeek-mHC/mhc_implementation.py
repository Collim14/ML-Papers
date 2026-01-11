import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
class Sinkhorn:
    @staticmethod
    def apply(login, iter = 20, eps = 1e-6):
        M = torch.exp(login)
        for _ in range(iter):
            M = M / (M.sum(dim = -1, keepdim = True)+eps)
            M= M / (M.sum(dim = -2, keepdim = True)+eps)
        return M
    @staticmethod
    def applylog(login, iter = 20, eps = 1e-6):
        logup = torch.zeros((login.shape[-1],))
        r = torch.zeros((login.shape[:-1]))
        c = torch.zeros((login.shape[:-1]))
        for _ in range(iter):
            r = logup - torch.logsumexp(login + r.unsqueeze(-2), dim=-1)
            c = logup - torch.logsumexp(login + c.unsqueeze(-1), dim=-2)

        return torch.exp(login +r.unsqueeze(-1)+c.unsqueeze(-2))
    


class mHCModule(nn.Module):
    def __init__(self, n_streams,dim, device=None):
        super().__init__()
        self.n_streams = n_streams
        self.device = device
        self.rms = nn.RMSNorm(n_streams * dim)

        self.alpha_pre = nn.Parameter(torch.tensor(0.001))
        self.alpha_post = nn.Parameter(torch.tensor(0.001))
        self.alpha_res = nn.Parameter(torch.tensor(0.001))

        self.phi_pre = nn.Linear(n_streams*dim, n_streams, bias = False)
        self.phi_post = nn.Linear(n_streams*dim, n_streams, bias = False)
        self.phi_res = nn.Linear(n_streams*dim, n_streams*n_streams, bias = False)

        self.b_pre = nn.Parameter(torch.zeros(n_streams))
        self.b_post = nn.Parameter(torch.zeros(n_streams))
        self.b_res = nn.Parameter(torch.zeros(n_streams*n_streams))

    def forward(self,x):
       
        b, s, n, d = x.shape
        x_flat = x.reshape(b, s, -1)
        x_norm = self.rms(x_flat)

        h_pre = self.alpha_pre *self.phi_pre(x_norm) + self.b_pre
        h_post = self.alpha_post*self.phi_post(x_norm) + self.b_post
        temp = self.phi_res(x_norm).reshape(b,s, n,n)
        h_res = self.alpha_res*temp +self.b_res.reshape(1,1,n,n)

        h_pre = torch.sigmoid(h_pre)
        h_post = 2*torch.sigmoid(h_post)
        h_res = Sinkhorn.applylog(h_res)

        resStream = torch.matmul(h_res, x)
        layIn = torch.einsum('bsn, bsnd -> bsd',h_pre, x)

        def closure(layerOut):
            hout = torch.einsum('bsn, bsd -> bsnd', h_post, layerOut)
            return resStream + hout
        return layIn, closure



