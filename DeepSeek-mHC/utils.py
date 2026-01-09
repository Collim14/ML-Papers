import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import requests
import os

class Tracker:
    def __init__(self):
        self.losses = []
        self.grad_total = []
        self.grad_mixing = [] 

    def update(self, loss, model, mhc=False):
        self.losses.append(loss)
        
        t_norm = 0.0
        m_norm = 0.0
        
        for name, p in model.named_parameters():
            if p.grad is not None:
                gn = p.grad.data.norm(2).item()
                t_norm += gn ** 2
                if mhc and any(x in name for x in ['alpha', 'phi', 'b_']):
                    m_norm += gn ** 2
                    
        self.grad_total.append(t_norm ** 0.5)
        self.grad_mixing.append(m_norm ** 0.5 if mhc else 0.0)

    def to_dict(self):
        return {
            "losses": self.losses,
            "grad_total": self.grad_total,
            "grad_mixing": self.grad_mixing
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