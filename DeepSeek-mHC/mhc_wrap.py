import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np
from mhc_implementation import Sinkhorn
import torch.nn.functional as F

@torch.jit.script
def newton_schulz_autograd(M, steps: int = 5):
    #muon coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    norm = torch.linalg.matrix_norm(M, ord = 'fro', dim = (-2,-1),keepdim=True)
    X = M.div(norm + 1e-6)
    for _ in range(steps):
        X = X.contiguous()
        X2 = X.transpose(-1, -2) @ X
        X3 = X@X2
        X5 = X3@X2
        X = b*X3 + c*X5 +a*X
    return X



class mHCWrapper(nn.Module):
    def __init__(self, n_streams,module, inpdim, outdim = None, stride=1, device=None, useNS = True):
        super().__init__()
        self.n_streams = n_streams
        self.device = device
        self.module = module
        self.inpdim = inpdim
        self.outdim = outdim if outdim is not None else inpdim
        self.stride = stride
        self.useNS =useNS

        self.is2d = isinstance(module, (nn.Conv2d, nn.Sequential)) and \
                     any(isinstance(m, nn.Conv2d) for m in module) if isinstance(module, nn.Sequential) else isinstance(module, nn.Conv2d)
        normdim = self.inpdim*self.n_streams

        if self.is2d:
            self.norm = nn.GroupNorm(self.n_streams, normdim)
            self.phi_pre = nn.Conv2d(normdim, n_streams, kernel_size = 1, bias = False)
            self.phi_post = nn.Conv2d(normdim, n_streams, kernel_size = 1,  bias = False)
            self.phi_res = nn.Conv2d(normdim, n_streams*n_streams,  kernel_size = 1, bias = False)
            bias_shape = (self.n_streams,1,1)
            res_bias_shape = (self.n_streams*self.n_streams, 1, 1)
        else:
            self.norm = nn.RMSNorm(normdim)
            self.phi_pre = nn.Linear(normdim, n_streams, bias = False)
            self.phi_post = nn.Linear(normdim, n_streams, bias = False)
            self.phi_res = nn.Linear(normdim, n_streams*n_streams, bias = False)
            bias_shape = (self.n_streams,)
            res_bias_shape = (self.n_streams*self.n_streams,)

        self.alpha_pre = nn.Parameter(torch.tensor(0.001))
        self.alpha_post = nn.Parameter(torch.tensor(0.001))
        self.alpha_res = nn.Parameter(torch.tensor(0.001))

        self.b_pre = nn.Parameter(torch.zeros(bias_shape))
        self.b_post = nn.Parameter(torch.zeros(bias_shape))
        self.b_res = nn.Parameter(torch.zeros(res_bias_shape))
        if self.useNS:
            self.conditioner = newton_schulz_autograd
        else:
            self.conditioner = Sinkhorn.applylog

        self.shortcut = None
        if self.inpdim != self.outdim or self.stride > 1:
            if self.is2d:
                self.shortcut = nn.Conv2d(self.inpdim*self.n_streams, self.outdim*self.n_streams, kernel_size = 1, stride = self.stride,groups=self.n_streams, bias =False)
            else:
                self.shortcut = nn.Linear(self.inpdim*self.n_streams, self.outdim*self.n_streams, bias = False)


    def forward(self,x):
        if self.is2d:
            b,n,c,h,w = x.shape
            x_flat = x.reshape(b,n*c,h,w).contiguous()
            #final shape of (b, n*c,h,w)
        else:
       
            b, s, n, d = x.shape
            x_flat = x.reshape(b, s, -1).contiguous()
            #final shape of (b, s, n*d)
        
        x_norm = self.norm(x_flat)

        #Now 2d it is (b,n,h,w), and otherwise (b, s, n)
        h_pre = self.alpha_pre *self.phi_pre(x_norm) + self.b_pre
        #Now 2d it is (b,n,h,w), and otherwise (b, s, n)
        h_post = self.alpha_post*self.phi_post(x_norm) + self.b_post
        #Now 2d it is (b,n*n,h,w), and otherwise (b, s, n*n)
        init_res = self.alpha_res*self.phi_res(x_norm) +self.b_res

        h_pre = torch.sigmoid(h_pre)
        h_post = 2*torch.sigmoid(h_post)

        if self.is2d:
            # (B, n*n, h,w) input, shaped to (b, n, n, h, w) but needs (b,h,w,n,n) for sinkhorn
            h_res = init_res.reshape(b,n,n,h,w).permute(0,3,4,1,2).contiguous()
            #h_res = Sinkhorn.applylog(h_res)
            h_res = self.conditioner(h_res)
            #Now we need layer inputs
            layIn = (x*h_pre.unsqueeze(2)).sum(dim=1).contiguous()
        else:
            #(B, s, n*n) input, we want (b, s, n, n)
            h_res = init_res.reshape(b,s,n,n).contiguous()
           # h_res = Sinkhorn.applylog(h_res)
            h_res = self.conditioner(h_res)
            layIn = torch.einsum('bsn, bsnd -> bsd',h_pre, x).contiguous()

        layOut = self.module(layIn)

        if self.is2d:
            #Because of matmul looking at last two dims, we need to permute x to multiply across matrices of dims
            #(n,n)x(n,c)
            xp = x.permute(0,3,4,1,2).contiguous()
            #permute back once calculated
            resStream = torch.matmul(h_res,xp).permute(0,3,4,1,2).contiguous()
            if self.shortcut:
                resStream = self.shortcut(resStream.reshape(b,-1,h,w)).contiguous()
                resStream = resStream.reshape(b, n, -1, layOut.shape[2], layOut.shape[3])
                if self.stride > 1:
                    h_post = F.avg_pool2d(h_post, self.stride, self.stride)
            return (resStream +layOut.unsqueeze(1)* h_post.unsqueeze(2)).contiguous()
                
            
        else:
            resStream = torch.matmul(h_res,x).contiguous()
            if self.shortcut:
                resStream = self.shortcut(resStream.reshape(b,s,-1))
                resStream = resStream.reshape(b,s,n,-1).contiguous()
            out = torch.einsum('bsd, bsn -> bsnd', layOut, h_post)
            return (resStream +out).contiguous()
        
class MHCEntry(nn.Module):
    def __init__(self, n): 
        super().__init__()
        self.n = n
    def forward(self, x):
        if x.dim() == 4:
            return x.unsqueeze(1).repeat(1, self.n, 1, 1, 1)
            
        elif x.dim() == 3:
            return x.unsqueeze(2).repeat(1, 1, self.n, 1)
        
        else:
            raise ValueError(f"MHCEntry expects 3D or 4D input, got {x.dim()}D")

class MHCExit(nn.Module):
    def forward(self, x_stream):
        if x_stream.dim() == 5:
            return x_stream.mean(dim=1)
        elif x_stream.dim() == 4:
            return x_stream.mean(dim=2)
            
        else:
             raise ValueError(f"MHCExit expects 4D or 5D input, got {x_stream.dim()}D")