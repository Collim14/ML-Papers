import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import requests
import os
import torch.nn as nn
from NanoGPT_mhc import Block, StandardResidual
from torch.nn import functional as F
import re

class Sinkhorn:
    @staticmethod
    def apply(login, iter = 20, eps = 1e-6):
        M = torch.exp(login -login.max())
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

class Tracker:
    def __init__(self,wrapper):
        self.losses = []
        self.grad_total = []
        self.grad_mixing = [] 
        self.lhres_mean = []
        self.lhres_max = []
        self.entropy = []
        self.spectral_norms = {}
        self.wrapper = wrapper

        self._hooks_registered = False
        self.layer_inputs = {}
        self.hooks = []
        self.pi_vectors = {}

    def _register_hooks(self, model):
        if self._hooks_registered == True: 
            return
        def get_hook(name):
            def hook(module,args,output):
                x = args[0].detach()
                x.requires_grad = True
                self.layer_inputs[name] = x
            return hook
        for name, module in model.named_modules():
            if isinstance(module, (self.wrapper,StandardResidual)):
                h = module.register_forward_hook(get_hook(name))
                self.hooks.append(h)
        self._hooks_registered = True
        print("Hooks registered")

    def metrics(self,tensor):
        if tensor.min() < 0:
            probs = tensor.pow(2)
        else:
            probs = tensor
        probs = probs.clamp(min=1e-8)
        ent_elements = torch.special.entr(probs) 
        entropy = ent_elements.sum(dim=(-2, -1))
        sigma = torch.linalg.matrix_norm(tensor, ord=2)
        return sigma.mean().item(),entropy.mean().item()

    def jsn(self, name, module, x, n_iters=1, eps=1e-4):
        if name in self.pi_vectors and x.shape != self.pi_vectors[name].shape:
            del self.pi_vectors[name]
        if name not in self.pi_vectors:
            v = torch.randn_like(x)
            v = F.normalize(v.reshape(-1), dim=0).reshape(x.shape)
            self.pi_vectors[name] = v
        else:
            v = self.pi_vectors[name]

        v1 = v.detach()
        sigma = 0.0
        for _ in range(n_iters):
            with torch.no_grad():
                yn = module(x)
                yp = module(x + eps * v1)
                
            u = (yp - yn) / eps
            
            with torch.enable_grad():
                output = module(x)
                v2 = torch.autograd.grad(
                    outputs=output,
                    inputs=x,
                    grad_outputs=u,
                    retain_graph=False,
                    create_graph=False
                )[0]
            v1 = F.normalize(v2.reshape(-1), dim=0).reshape(x.shape)
            
            sigma = u.reshape(-1).norm(2).item()

        self.pi_vectors[name] = v1
        return sigma
    
    def rename(self,name):
        suffix = {'.attn_wrapper':'A', '.mlp_wrapper':'M'}
        pattern = r"blocks\.(\d+)(\.attn_wrapper|\.mlp_wrapper)"
        def replace(match):
            num =match.group(1)
            suff = match.group(2)
            newsuff = suffix.get(suff,suff)
            return f"Block_{num}_{newsuff}"
        return re.sub(pattern,replace,name)


    def update(self, loss, model, mhc=False):
        self.losses.append(loss)
        
        t_norm = 0.0
        m_norm = 0.0
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                gn = p.grad.data.norm(2).item()
                t_norm += gn ** 2
                if mhc and any(x in name for x in ['alpha', 'phi', 'b_', 'h_res', 'h_pre', 'h_post']):
                    m_norm += gn ** 2
        lhreslayer = []
        ents = []
        self.grad_total.append(t_norm ** 0.5)
        self.grad_mixing.append(m_norm ** 0.5 if mhc else 0.0)
        for module in model.modules():
            if hasattr(module, 'lhres') and module.lhres is not None:
                sigma, entropy = self.metrics(module.lhres)
                lhreslayer.append(sigma)
                ents.append(entropy)
                module.lhres = None
        if len(lhreslayer)>0:
            self.lhres_max.append(max(lhreslayer))
            self.lhres_mean.append(sum(lhreslayer)/len(lhreslayer))
            self.entropy.append(sum(ents)/len(ents))
        else:
            self.lhres_max.append(0.0)
            self.lhres_mean.append(0.0)
            self.entropy.append(0.0)
        for name, x in self.layer_inputs.items():
            module = dict(model.named_modules())[name]
            
            sn = self.jsn(name, module, x)
            displayname = self.rename(name)
            
            if displayname not in self.spectral_norms:
                self.spectral_norms[displayname] = []
            self.spectral_norms[displayname].append(sn)
        self.layer_inputs.clear()

        
                

    def to_dict(self):
        return {
            "losses": self.losses,
            "grad_total": self.grad_total,
            "grad_mixing": self.grad_mixing,
            "spectral_norms":self.spectral_norms
        }

def mnist():
    t = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.13,), (0.3,))
    ])
    BATCH_SIZE = 128
    
    train_d = datasets.MNIST('./data', train=True, download=True, transform=t)
    test_d = datasets.MNIST('./data', train=False, download=True, transform=t)
    
    train_loader = DataLoader(train_d, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_d, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def tinyshakespeare():
    if not os.path.exists('input.txt'):
        r = requests.get('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
        with open('input.txt', 'w') as f: f.write(r.text)
    
    with open('input.txt', 'r') as f: 
        text = f.read()[:50000]
    
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    BATCH_SIZE = 32
    BLOCK_SIZE = 64
    
    class DS(Dataset):
        def __init__(self, d, block_size): 
            self.d = d
            self.block_size = block_size
        def __len__(self): 
            return len(self.d) - self.block_size
        def __getitem__(self, i): 
            return self.d[i : i + self.block_size], self.d[i + 1 : i + self.block_size + 1]
    
    train_loader = DataLoader(DS(train_data, BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(DS(val_data, BLOCK_SIZE), batch_size=BATCH_SIZE, shuffle=False)
    
    return (train_loader, test_loader), vocab_size
