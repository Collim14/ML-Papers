import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils import Tracker, mnist, tinyshakespeare
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import requests
import matplotlib.pyplot as plt
import numpy as np
from NanoGPT_mhc import NanoGPT
from Conv_mhc import MNISTModel

DEVICE = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = "results"


def run(task, train_loader, test_loader, vocab=None):
    variants = {
        'Baseline': 'base',
        'mHC (NS)': 'true',
        'mHC (SH)': 'false'
    }
    
    results = {}
    print(f"\nBenchmark: {task}")
    
    for name, mode in variants.items():
        print(f"\nTraining {name}")
        
        if task == "MNIST":
            if mode == 'base': 
                model = MNISTModel(mhc=False)
            else: 
                model = MNISTModel(mhc=True, useNS=(mode=='true'))
        else:
            if mode == 'base': 
                model = NanoGPT(vocab, dim=64, n_streams=4, mhc=False)
            else: 
                model = NanoGPT(vocab, dim=64, n_streams=4, mhc=True, useNS=(mode=='true'))
            
        model = model.to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        trk = Tracker()

        model.train()
        if task == "MNIST":
            EPOCHS = 3
            for epoch in range(EPOCHS):
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
                for x, y in pbar:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    pred = model(x)
                    loss = crit(pred, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    acc = (pred.argmax(1) == y).float().mean().item()
                    trk.update(loss.item(), model, mhc=(mode!='base'))
                    pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")
        else:
            ITERATIONS = 500
            it = iter(train_loader)
            for _ in tqdm(range(ITERATIONS), desc="Training Shakespeare"):
                try: 
                    x, y = next(it)
                except StopIteration: 
                    it = iter(train_loader)
                    x, y = next(it)
                
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                pred = model(x)
                
                B, T, C = pred.shape
                loss = crit(pred.view(B*T, C), y.view(B*T))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                acc = (pred.argmax(dim=-1) == y).float().mean().item()
                trk.update(loss.item(), model, mhc=(mode!='base'))
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")

        model.eval()
        total_test_loss = 0
        correct_preds = 0
        total_samples = 0
        
        print(f"Evaluating {name} on test set...")
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                
                if task != "MNIST":
                    B, T, C = pred.shape
                    loss = crit(pred.view(B*T, C), y.view(B*T))
                    total_test_loss += loss.item()
                    argmax_preds = pred.argmax(dim=-1)
                    correct_preds += (argmax_preds == y).sum().item()
                    total_samples += y.numel()
                else:
                    loss = crit(pred, y)
                    total_test_loss += loss.item()
                    argmax_preds = pred.argmax(dim=1)
                    correct_preds += (argmax_preds == y).sum().item()
                    total_samples += y.size(0)

        avg_test_loss = total_test_loss / len(test_loader)
        test_acc = 100.0 * correct_preds / total_samples
        print(f"Test Loss: {avg_test_loss:.4f} | Test Acc: {test_acc:.2f}%")

        results[name] = {
            'tracker': trk,
            'test_loss': avg_test_loss,
            'test_acc': test_acc
        }
        
        save_data = trk.to_dict()
        save_data.update({'test_loss': avg_test_loss, 'test_acc': test_acc})
        with open(f"{RESULTS_DIR}/{task}_{mode}.json", 'w') as f:
            json.dump(save_data, f)
            
    return results
def plot(task, res):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{task} Results", fontsize=16)
    
    cols = {'Baseline':'gray', 'mHC (NS)':'green', 'mHC (SH)':'red'}
    
    def smooth(d): return np.convolve(d, np.ones(5)/5, mode='valid') if len(d)>5 else d
    
    for name, trk in res.items():
        c = cols.get(name, 'blue')
        if len(trk.losses) > 10:
            ax[0].plot(smooth(trk.losses), label=name, color=c)
            ax[1].plot(smooth(trk.grad_total), label=name, color=c)
            if 'Baseline' not in name:
                ax[2].plot(smooth(trk.grad_mixing), label=name, color=c)
                
    ax[0].set_title("Training Loss")
    ax[0].legend()
    ax[1].set_title("Total Gradient Norm")
    ax[2].set_title("MHC Mixing Gradients")
    
    plt.savefig(f"{RESULTS_DIR}/{task}_plot.png")
    plt.show()

if __name__ == "__main__":
    dl,tl = mnist()
    res_m = run("MNIST", dl, tl)
    plot("MNIST", res_m)
    
    (dl, tl), v = tinyshakespeare()
    res_g = run("GPT", dl,tl, vocab=v)
    plot("GPT", res_g)