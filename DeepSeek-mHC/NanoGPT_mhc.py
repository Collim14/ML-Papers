import torch
import torch.nn as nn
from mhc_wrap import mHCWrapper, MHCEntry, MHCExit

class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x):
        B, T, C = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        return out
    
class Block(nn.Module):
    def __init__(self, dim, wrapper_func):
        super().__init__()
        attn_branch = nn.Sequential(
            nn.LayerNorm(dim),
            SelfAttention(dim, heads=4)
        )
        self.attn_wrapper = wrapper_func(attn_branch, dim)
        
        mlp_branch = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.mlp_wrapper = wrapper_func(mlp_branch, dim)

    def forward(self, x):
        x = self.attn_wrapper(x)
        x = self.mlp_wrapper(x)
        return x



class NanoGPT(nn.Module):
    def __init__(self, vocab_size, dim=64, n_streams=4, depth=2, block_size=256, mhc=True, useNS=False):
        super().__init__()
        self.mhc = mhc
        self.n_streams = n_streams if mhc else 1
        self.useNS = useNS
        self.block_size = block_size
        
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, block_size, dim) * 0.02)
        
        self.entry = MHCEntry(n_streams) if mhc else nn.Identity()
        
        self.blocks = nn.Sequential(*[
            Block(dim, self._wrap) for _ in range(depth)
        ])
        
        self.exit = MHCExit() if mhc else nn.Identity()
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def _wrap(self, module, dim):
        if not self.mhc:
            class StandardResidual(nn.Module):
                def __init__(self, mod): 
                    super().__init__()
                    self.mod = mod
                def forward(self, x): 
                    return x + self.mod(x)
            return StandardResidual(module)

        return mHCWrapper(n_streams=self.n_streams, module=module, inpdim=dim, useNS=self.useNS)

    def forward(self, idx):
        B, T = idx.shape
        if T > self.block_size:
             idx = idx[:, -self.block_size:]
             T = self.block_size

        x = self.token_embed(idx) + self.pos_embed[:, :T, :]
        x = self.entry(x)
        x = self.blocks(x)
        x = self.exit(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits