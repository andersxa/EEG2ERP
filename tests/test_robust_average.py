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
# Import robust averaging methods from algorithms package
from algorithms.robust_average import (
    leonowicz_predict_fn,
    kotowski_predict_fn,
    molina_fast_predict_fn,
    traditional_predict_fn
)

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
