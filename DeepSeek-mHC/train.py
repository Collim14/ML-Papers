import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import re
from utils import Tracker, mnist, tinyshakespeare
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from NanoGPT_mhc import NanoGPT
from Conv_mhc import MNISTModel
from datetime import datetime
from mhc_wrap import mHCWrapper
DEVICE = torch.device("cpu" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = "results"

def runs(task, train_loader, test_loader, vocab=None):
    variants = {
        'mHC (NS)': 'true',
        'mHC (SH)': 'false',
        'HC': None,
        'Baseline': 'base'
    }
    
    results = {}
    print(f"\nBenchmark: {task}")
    
    for name, mode in variants.items():
        print(f"\nTraining {name}")
        
        if task == "MNIST":
            if mode == 'base': 

                model = MNISTModel(mhc=False)
            elif mode == None:
                model = MNISTModel(mhc=True, useNS=None, static = True)
            else: 
                model = MNISTModel(mhc=True, useNS=(mode=='true'),static = True)
        else:
            if mode == 'base': 
                model = NanoGPT(vocab, dim=64,  n_streams=4,depth=24, mhc=False)
            elif mode == None:
                model = NanoGPT(vocab, dim=64, n_streams=4,depth = 24,  mhc=True, useNS=None, static = True)
            else: 
                model = NanoGPT(vocab, dim=64, n_streams=4,depth = 24,  mhc=True, useNS=(mode=='true'),static = True)
            
        model = model.to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        trk = Tracker(wrapper = mHCWrapper)
        trk._register_hooks(model)

        model.train()
        if task == "MNIST":
            EPOCHS = 1
            for epoch in range(EPOCHS):
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
                for x, y in pbar:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    pred = model(x)
                    loss = crit(pred, y)
                    loss.backward()
                    trk.update(loss.item(), model, mhc=(mode!='base'))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    acc = (pred.argmax(1) == y).float().mean().item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")
        else:
            ITERATIONS = 50
            it = iter(train_loader)
            pbar = tqdm(range(ITERATIONS), desc="Training Shakespeare")
            for _ in pbar:
                try: 
                    x, y = next(it)
                except StopIteration: 
                    it = iter(train_loader)
                    x, y = next(it)
                
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                pred = model(x)
                
                B, T, C = pred.shape
                loss = crit(pred.reshape(B*T, C), y.reshape(B*T))
                
                loss.backward()
                trk.update(loss.item(), model, mhc=(mode!='base'))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                acc = (pred.argmax(dim=-1) == y).float().mean().item()
                
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
                    loss = crit(pred.reshape(B*T, C), y.reshape(B*T))
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

        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        with open(f"{RESULTS_DIR}/{task}_{mode}_{str(current_time)}.json", 'w') as f:
            json.dump(save_data, f)
            
    return results


def run(task, train_loader, test_loader, vocab=None):
    variants = {
        'mHC (NS)': 'true',
        'mHC (SH)': 'false',
        'HC': None,
        'Baseline': 'base'
    }
    
    results = {}
    print(f"\nBenchmark: {task}")
    
    for name, mode in variants.items():
        print(f"\nTraining {name}")
        
        if task == "MNIST":
            if mode == 'base': 

                model = MNISTModel(mhc=False)
            elif mode == None:
                #model = MNISTModel(mhc=True, useNS=None)
                model = MNISTModel(mhc=True, useNS=None)
            else: 
                model = MNISTModel(mhc=True, useNS=(mode=='true'))
        else:
            if mode == 'base': 
                model = NanoGPT(vocab, dim=64,  n_streams=4,depth=24, mhc=False)
            elif mode == None:
                model = NanoGPT(vocab, dim=64, n_streams=4,depth = 24,  mhc=True, useNS=None)
            else: 
                model = NanoGPT(vocab, dim=64, n_streams=4,depth = 24,  mhc=True, useNS=(mode=='true'))
            
        model = model.to(DEVICE)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        trk = Tracker(wrapper = mHCWrapper)
        trk._register_hooks(model)

        model.train()
        if task == "MNIST":
            EPOCHS = 1
            for epoch in range(EPOCHS):
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
                for x, y in pbar:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    opt.zero_grad()
                    pred = model(x)
                    loss = crit(pred, y)
                    loss.backward()
                    trk.update(loss.item(), model, mhc=(mode!='base'))
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    acc = (pred.argmax(1) == y).float().mean().item()
                    pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.2%}")
        else:
            ITERATIONS = 50
            it = iter(train_loader)
            pbar = tqdm(range(ITERATIONS), desc="Training Shakespeare")
            for _ in pbar:
                try: 
                    x, y = next(it)
                except StopIteration: 
                    it = iter(train_loader)
                    x, y = next(it)
                
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                pred = model(x)
                
                B, T, C = pred.shape
                loss = crit(pred.reshape(B*T, C), y.reshape(B*T))
                
                loss.backward()
                trk.update(loss.item(), model, mhc=(mode!='base'))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                acc = (pred.argmax(dim=-1) == y).float().mean().item()
                
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
                    loss = crit(pred.reshape(B*T, C), y.reshape(B*T))
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

        current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        with open(f"{RESULTS_DIR}/{task}_{mode}_{str(current_time)}.json", 'w') as f:
            json.dump(save_data, f)
            
    return results


def plot(task, res):
    current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    fig_metrics, ax = plt.subplots(1, 5, figsize=(24, 5))
    fig_metrics.suptitle(f"{task} Training Metrics", fontsize=16)
    heatmap_storage = {} 
    cols = {'HC':'blue','Baseline':'gray', 'mHC (NS)':'green', 'mHC (SH)':'red'}
    
    def smooth(d): 
        return np.convolve(d, np.ones(5)/5, mode='valid') if len(d)>5 else d
    
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]

    for name, data in res.items():
        if isinstance(data, dict) and 'tracker' in data:
            trk = data['tracker']
        else:
            trk = data
        
        c = cols.get(name, 'purple')
        
        losses = getattr(trk, 'losses', [])
        if len(losses) > 10:
            ax[0].plot(smooth(losses), label=name, color=c)
            ax[1].plot(smooth(getattr(trk, 'grad_total', [])), color=c)
            
            if 'Baseline' not in name:
                ax[2].plot(smooth(getattr(trk, 'grad_mixing', [])), color=c)
                
            h_max = getattr(trk, 'lhres_max', [])
            if len(h_max) > 0:
                ax[3].plot(smooth(h_max), label=f"{name} (Max)", color=c)
                ax[3].plot(smooth(getattr(trk, 'lhres_mean', [])), color=c, linestyle='--', alpha=0.5)
            
            entropy = getattr(trk, 'entropy', [])
            if len(entropy) > 0:
                ax[4].plot(smooth(entropy), color=c)
        
        spec_data = getattr(trk, 'spectral_norms', {})
        if spec_data:
            layers = sorted(spec_data.keys(), key=natural_sort_key)
            matrix = np.array([spec_data[l] for l in layers])
            heatmap_storage[name] = {'matrix': matrix, 'layers': layers}

    ax[0].set_title("Training Loss")
    ax[0].legend()
    ax[1].set_title("Total Grad Norm")
    ax[1].set_yscale('log')
    ax[2].set_title("MHC Mixing Grads")
    ax[2].set_yscale('log')
    ax[3].set_title("H_res Norms")
    ax[4].set_title("Mixing Entropy")
    
    fig_metrics.tight_layout()
    metrics_filename = f"{RESULTS_DIR}/{task}_metrics_{current_time}.png"
    fig_metrics.savefig(metrics_filename)
    print(f"Saved metrics: {metrics_filename}")
    
    plt.close(fig_metrics)
    if heatmap_storage:
        all_values = []
        for data in heatmap_storage.values():
            all_values.append(data['matrix'].flatten())
        
        if all_values:
            combined_data = np.concatenate(all_values)
            g_min = combined_data.min()
            g_max = combined_data.max()
            print(f"Heatmap Shared Scale: {g_min:.4f} to {g_max:.4f}")

            for name, data in heatmap_storage.items():
                matrix = data['matrix']
                layers = data['layers']

                fig_heatmap = plt.figure(figsize=(10, 8))
                
                sns.heatmap(matrix, cmap='magma', yticklabels=layers, 
                            vmin=g_min, vmax=g_max)
                
                plt.title(f"Spectral Norm - {name}\n(Scale: {g_min:.2f} - {g_max:.2f})")
                plt.xlabel("Iterations")
                plt.ylabel("Block Depth")
                hm_filename = f"{RESULTS_DIR}/{name}_spectral_{current_time}.png"
                plt.savefig(hm_filename, bbox_inches='tight')
                plt.close(fig_heatmap)
                print(f"Saved heatmap: {hm_filename}")

    

if __name__ == "__main__":
    dl,tl = mnist()
    res_m = run("MNIST", dl, tl)
    plot("MNIST", res_m)
    
    (dl, tl), v = tinyshakespeare()
    res_g = run("GPT", dl,tl, vocab=v)
    plot("GPT", res_g)
    

    dl,tl = mnist()
    res_m = runs("MNIST", dl, tl)
    plot("MNIST", res_m)
    
    (dl, tl), v = tinyshakespeare()
    res_g = runs("GPT", dl,tl, vocab=v)
    plot("GPT", res_g)