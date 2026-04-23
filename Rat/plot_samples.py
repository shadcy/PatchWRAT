import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from benchmark_pwsa_vs_wrat import PatchWRAT, get_loaders, DEVICE, train_wrat, LR
import torch.optim as optim

def train_and_plot():
    SEQ_LEN = 336
    PRED_LEN = 96
    
    print("Loading data...")
    tr_loader, va_loader, te_loader = get_loaders(PRED_LEN)
    
    model = PatchWRAT(SEQ_LEN, PRED_LEN).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    
    # Train for more epochs to see convergence
    epochs = 30
    print(f"Training PatchWRAT for {epochs} epochs to generate converged samples...")
    for ep in range(epochs):
        loss = train_wrat(model, tr_loader, opt)
        print(f"Epoch {ep+1}/{epochs} | Training Loss: {loss:.5f}")
        
    print("Evaluating and plotting samples...")
    model.eval()
    bx, by = next(iter(te_loader))
    bx, by = bx.to(DEVICE), by.to(DEVICE)
    with torch.no_grad():
        preds, _, _, _ = model(bx)
        
    preds = preds.cpu().numpy()
    by = by.cpu().numpy()
    bx = bx.cpu().numpy()
    
    # Plot samples
    fig, axes = plt.subplots(4, 1, figsize=(12, 14))
    
    # We will plot the past (input) as well as future (ground truth and prediction)
    for i in range(4):
        sample_idx = i
        channel = 0 # Selecting the first variate (e.g., High Use / Oil Temp)
        
        past = bx[sample_idx, channel, :]
        truth = by[sample_idx, channel, :]
        pred = preds[sample_idx, channel, :]
        
        x_past = np.arange(-SEQ_LEN, 0)
        x_future = np.arange(0, PRED_LEN)
        
        axes[i].plot(x_past, past, label='Input (Past)', color='gray')
        axes[i].plot(x_future, truth, label='Ground Truth', color='blue')
        axes[i].plot(x_future, pred, label='Predicted (PatchWRAT)', color='red', linestyle='--')
        
        axes[i].set_title(f'Test Sample {sample_idx+1} | Horizon = {PRED_LEN}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('samples_plot.png', dpi=150)
    print("Saved samples_plot.png successfully.")
    
    # Textual sample output
    print("\nSAMPLE PREDICTIONS (First 10 steps of Sample 1, Channel 0):")
    print("-" * 60)
    print(f"{'Step':>5} | {'Ground Truth':>12} | {'Predicted':>12}")
    print("-" * 60)
    for t in range(10):
        print(f"{t:>5} | {by[0, 0, t]:>12.4f} | {preds[0, 0, t]:>12.4f}")
    print("-" * 60)
    
    # Extract and output learned parameters
    print("\n" + "="*50)
    print("LEARNED PARAMETERS:")
    print("="*50)
    print(f"Learned Tau threshold (sigmoid of raw_tau): {torch.sigmoid(model.wrat.raw_tau).item():.4f}")
    
    # DWT filters shape is (out_ch, in_ch, filter_len)
    # They are just 1D convolutions
    print("\nLearned DWT Low-pass Filters (h) - First 5 out of", model.dwt.h.shape[0], "channels:")
    print(model.dwt.h[:5].squeeze(1).detach().cpu().numpy())
    
    print("\nLearned DWT High-pass Filters (g) - First 5 out of", model.dwt.g.shape[0], "channels:")
    print(model.dwt.g[:5].squeeze(1).detach().cpu().numpy())
    print("="*50)

if __name__ == '__main__':
    train_and_plot()
