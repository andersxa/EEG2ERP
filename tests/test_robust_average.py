#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import torch.nn.functional as F

# Create a random tensor
mean = torch.sin(torch.linspace(0, 2 * 3.1415, 256)).unsqueeze(0)
std = 0.2
x1 = torch.randn(5, 256) * std + mean
x2 = torch.randn(2, 256) * std - mean
x3 = torch.randn(1, 256)
x = torch.cat([x1, x2, x3], dim=0).unsqueeze(0)
# Plot the data
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].set_title('Signal 1 sin(t) with Noise')
ax[0].plot(x1.T.cpu().numpy())
ax[1].set_title('Signal 2 -sin(t) with Noise')
ax[1].plot(x2.T.cpu().numpy())
ax[2].set_title('Signal 3 Noise')
ax[2].plot(x3.T.cpu().numpy())
ax[0].set_xlabel('Time')
ax[0].set_ylabel('Value')
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Value')
ax[2].set_xlabel('Time')
ax[2].set_ylabel('Value')
plt.suptitle('Signals used in example')
plt.tight_layout()
plt.savefig("figures/robust_average_signals.png")

#%%
from dtaidistance import dtw
import numpy as np

def leonowicz_predict_fn(x: torch.Tensor):
    # In shape: (repeats, trials, ...)
    k = 0.1
    v = 0
    N = x.size(1)
    #std = x.std(dim=(-2, -1)) # shape: (repeats, trials)
    # Sort by amplitude
    sorted_indices = torch.argsort(x, dim=1)
    inverse_indices = torch.argsort(sorted_indices, dim=1)
    first_half = (inverse_indices < N//2)
    second_half = (inverse_indices >= N//2)
    inverse_indices = inverse_indices.float()
    weights = first_half * (torch.tanh(k * (inverse_indices + 1)) - v) + second_half * (-torch.tanh(k * (inverse_indices - N)) - v)
    #weights[weights < 0] = 0
    weights /= weights.sum(dim=1, keepdim=True)
    x_pred = (x * weights).sum(dim=1)
    return x_pred

def kotowski_predict_fn(x: torch.Tensor):
    # In shape: (repeats, trials, timepoints)

    # Hyperparameters
    tol = 1e-8
    m = 2 #2
    c = 100 #100
    norm_p = 2 #2
    exponent = 1 / (1-m)
    N = x.size(1)
    zero_val = 1 / (c * N)
    iterations = 1000

    # Step 1: Initialize v0 with preferably the traditional average
    v = x.mean(dim=1, keepdim=True)
    # Step 2: Initialize w0 using eq. 3 with improvements from 3.2.1 and 3.2.2
    w_i = (torch.norm(v-x, p=norm_p, dim=-1, keepdim=True)+tol) ** exponent
    w_i = w_i / w_i.sum(dim=1, keepdim=True)
    zero_w = w_i < zero_val
    w_i = torch.where(zero_w, torch.tensor(0.0, device=w_i.device), w_i)
    w_i = w_i / w_i.sum(dim=1, keepdim=True)
    # Step 3: Update the averaged signal v0 using eq. 5
    w_e = w_i ** m
    v = (x * w_e).sum(dim=1, keepdim=True) / w_e.sum(dim=1, keepdim=True)
    for i in range(iterations):
        # Step 4: Calculate the Pearson correlations between the current v and each epoch. Rescale to (0, 1)
        v_mean = v.mean(dim=-1, keepdim=True)
        x_mean = x.mean(dim=-1, keepdim=True)

        v_centered = v - v_mean
        x_centered = x - x_mean

        v_ss = (v_centered ** 2).sum(dim=-1, keepdim=True)
        x_ss = (x_centered ** 2).sum(dim=-1, keepdim=True)

        dot_product = (v_centered * x_centered).sum(dim=-1, keepdim=True)
        u = dot_product / (torch.sqrt(v_ss * x_ss))
        u_scaled = (u + 1) / 2

        # Step 5: Calculate the weights wl using eq. 3 with improvements from 3.2.1
        w_i = (torch.norm(v-x, p=norm_p, dim=-1, keepdim=True)+tol) ** exponent
        w_i = w_i / w_i.sum(dim=1, keepdim=True)
        # Step 6: Update the weights wl by multiplying sample-wise by ul.
        w_i = w_i * u_scaled
        # Step 7: Update the vector wl using eq. 4.
        w_i = w_i / w_i.sum(dim=1, keepdim=True)
        #Step 8: Update the weights wl using the improvements from 3.2.2
        zero_w = w_i < zero_val
        w_i = torch.where(zero_w, torch.tensor(0.0, device=w_i.device), w_i)
        #Step 9: Update the averaged signal vl using eq. 5.
        v_new = (x * w_i).sum(dim=1, keepdim=True) / w_i.sum(dim=1, keepdim=True)
        #Step 9: If the relative change is larger than epsilon then keep going, else break.
        change = torch.norm(v_new - v)
        v = v_new
        if change < tol:
            break
    x_pred = v.squeeze(1)
    return x_pred
from dtaidistance import dtw
from joblib import Parallel, delayed
def process_repeat(s_r, s_i, K):
    s_warped = []
    for i in range(K):
        _, paths = dtw.warping_paths_fast(s_r, s_i[i], use_c=True)
        path = dtw.best_path(paths)
        s_warp, path = dtw.warp(s_r, s_i[i], path, use_c=True)
        s_warped.append(s_warp)
    return s_warped
def molina_fast_predict_fn(x: torch.Tensor):
    # In shape: (repeats, trials, channels, timepoints)
    r = x.mean(dim=1).cpu().numpy().astype(float)
    x_n = x.cpu().numpy().astype(float)
    
    reps = Parallel(n_jobs=-1)(delayed(process_repeat)(r[repeat].copy(), x_n[repeat].copy(), x.size(1)) for repeat in range(x.size(0)))
    
    x_pred = torch.stack([torch.tensor(v, device=x.device, dtype=x.dtype).mean(dim=0) for v in reps])
    return x_pred

def traditional_predict_fn(x: torch.Tensor):
    v = x.mean(dim=1)
    return v

#%%
# Test the robust average methods
to_plot = []
to_plot.append((leonowicz_predict_fn(x), 'Leonowicz et al.'))
to_plot.append((kotowski_predict_fn(x), 'Kotowski et al.'))
to_plot.append((molina_fast_predict_fn(x), 'Molina et al.'))
to_plot.append((traditional_predict_fn(x), 'Traditional'))
fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=180)
for v, name in to_plot:
    print(f'{name} method:')
    print((mean-v).square().mean())
    print('\n')
    ax.plot(v.squeeze().cpu().numpy(), label=name)
ax.plot(mean.squeeze().cpu().numpy(), label='Signal 1', color='black', linestyle='--')
ax.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Averaging methods on toy data')
plt.tight_layout()
plt.savefig("figures/robust_average_results.png")
#%%
