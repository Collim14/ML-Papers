import torch
import torch.nn as nn
from mhc_wrap import mHCWrapper, MHCEntry, MHCExit

class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):
        B, T, C = x.shape
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        out, _ = self.attn(x, x, x, attn_mask=mask, is_causal=True)
        return out


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, dim=64, n_streams=4, depth=2, mhc=True, useNS = False):
        super().__init__()
        self.mhc = mhc
        self.n_streams = n_streams if mhc else 1
        self.useNS = useNS
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 128, dim))
        
        self.entry = MHCEntry(n_streams) if mhc else nn.Identity()
        
        layers = []
        for _ in range(depth):
            attn = SelfAttention(dim, heads=4)
            layers.append(self._wrap(attn, dim))
            mlp = nn.Sequential(
                nn.Linear(dim,dim*4),
                nn.GELU(),
                nn.Linear(dim*4,dim))
            layers.append(self._wrap(mlp, dim))
        self.backbone = nn.Sequential(*layers)
        self.exit = MHCExit() if mhc else nn.Identity()
        self.head = nn.Linear(dim, vocab_size)
    def _wrap(self, module, dim):
        if not self.mhc:
            return module
        return mHCWrapper(n_streams = self.n_streams, module = module, inpdim=dim, useNS=self.useNS)
    def forward(self, idx):
        B, T = idx.shape
        
        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        x = self.entry(x)
        x = self.backbone(x)
        x = self.exit(x)
        
        logits = self.head(x)
        return logits
