import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from mhc_wrap import mHCWrapper, MHCEntry, MHCExit

class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config, wrapper_func, depth = None):
        super().__init__()
        
        attn_branch = nn.Sequential(
            LayerNorm(config.n_embd, bias=config.bias),
            CausalSelfAttention(config)
        )
        self.attn_wrapper = wrapper_func(attn_branch, config.n_embd, depth = depth)

        mlp_branch = nn.Sequential(
            LayerNorm(config.n_embd, bias=config.bias),
            MLP(config)
        )
        self.mlp_wrapper = wrapper_func(mlp_branch, config.n_embd, depth = depth)

    def forward(self, x):
        x = self.attn_wrapper(x)
        x = self.mlp_wrapper(x)
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class StandardResidual(nn.Module):
    def __init__(self, mod): 
        super().__init__()
        self.mod = mod
    def forward(self, x): 
        return x + self.mod(x)

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, dim=64, n_streams=4, depth=2, block_size=256, mhc=True, useNS=False, static=False, n_head=4, dropout=0.0):
        super().__init__()
        
        self.mhc = mhc
        self.n_streams = n_streams if mhc else 1
        self.useNS = useNS
        self.static = static
        self.block_size = block_size

        self.config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=depth,
            n_head=n_head,
            n_embd=dim,
            dropout=dropout,
            bias=True
        )

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(block_size, dim)
        self.drop = nn.Dropout(dropout)
        self.entry = MHCEntry(self.n_streams) if mhc else nn.Identity()
        self.blocks = nn.ModuleList([
            Block(self.config, self._wrap, depth = i+1) for i in range(depth)
        ])
        
        self.exit = MHCExit() if mhc else nn.Identity()
        
        self.ln_f = LayerNorm(dim, bias=True)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        self.token_embed.weight = self.head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * depth))

    def _wrap(self, module, dim, depth = None):
       
        if not self.mhc:
            
            return StandardResidual(module)

        return mHCWrapper(
            n_streams=self.n_streams, 
            module=module, 
            inpdim=dim, 
            useNS=self.useNS, 
            static=self.static, depth = depth
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        if t > self.block_size:
             idx = idx[:, -self.block_size:]
             t = self.block_size

        pos = torch.arange(0, t, dtype=torch.long, device=device) 
        tok_emb = self.token_embed(idx)
        pos_emb = self.pos_embed(pos)
        x = self.drop(tok_emb + pos_emb)
        
        x = self.entry(x)
        for block in self.blocks:
            x = block(x)
        x = self.exit(x)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx