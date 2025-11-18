from model import ERPUNet, ERPAE, CSLPAE
import torch
from sklearn.decomposition import PCA
from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

task_names = {
    0: 'ERN/Incorrect',
    1: 'ERN/Correct',
    2: 'LRP/Contralateral',
    3: 'LRP/Ipsilateral',
    4: 'MMN/Deviants',
    5: 'MMN/Standards',
    6: 'N2pc/Contralateral',
    7: 'N2pc/Ipsilateral',
    8: 'N170/Faces',
    9: 'N170/Cars',
    10: 'N400/Unrelated',
    11: 'N400/Related',
    12: 'P3/Rare',
    13: 'P3/Frequent',
}

def load_bootstrap_plot(path, model):
    results = torch.load(f"{path}Models/{model}_test_results.pt", weights_only=False)
    return results['plot']

def plot_bootstrap_curve(epoch, path, subjects, tasks, repeats, steps_dict, *loss_dicts, split='dev'):
    fig, axs = plt.subplots(len(subjects), len(tasks), figsize=(3.2*len(tasks), 3*len(subjects)), dpi=80)
    averaging_loss = torch.load(f"{path}Loss/averaging_{split}_{repeats}.pt", weights_only=False)
    avg_loss_dict = averaging_loss['loss']
    loss_dicts = [("avg", avg_loss_dict), *loss_dicts]
    total_losses = defaultdict(float)
    num_total = 0
    for i, s in enumerate(subjects):
        for j, t in enumerate(tasks):
            ax = axs[i, j]
            model_steps = steps_dict[s][t]
            for name, loss_dict in loss_dicts:
                model_cur_loss = loss_dict[s][t]
                if isinstance(model_cur_loss, torch.Tensor):
                    model_cur_loss = model_cur_loss.cpu().numpy()
                if name == 'avg':
                    ax.plot(model_steps, model_cur_loss, label=name, color='black')
                else:
                    ax.plot(model_steps, model_cur_loss, label=name)
                total_losses[name] += model_cur_loss.mean()
            ax.axvline(x=5, color='black', linestyle='--')
            ax.set_yscale('log')
            ax.set_title(f"S: {s}, T: {task_names[t]}")
            ax.set_xlabel('#Trials')
            ax.set_ylabel('MSE')
            num_total += 1
    axs.flatten()[0].legend()
    for name, total_loss in total_losses.items():
        total_losses[name] = 1e12 * total_loss / num_total
    loss_string = ', '.join([f"{k}: {v:.3g}" for k, v in total_losses.items()])
    fig.suptitle(f"Epoch: {epoch:.1f}, {loss_string}")
    fig.tight_layout()
    return fig, axs

def load_r_squared(path, model):
    results = torch.load(f"{path}Models/{model}_test_results.pt", weights_only=False)
    df_rsquared = pd.DataFrame([(s, task_names[t], v.item() if isinstance(v, torch.Tensor) else v, model) for s in results['r_squared'].keys() for t, v in results['r_squared'][s].items()], columns=['Subject', 'Task', 'R^2', 'Method'])
    return df_rsquared

def plot_r_squared_boxplots(epoch, path, repeats, *dfs, split='dev'):
    averaging_100_loss = torch.load(f"{path}Loss/averaging_100_{split}_{repeats}.pt", weights_only=False)
    averaging_loss = torch.load(f"{path}Loss/averaging_{split}_{repeats}.pt", weights_only=False)
        
    df = pd.concat([averaging_loss['r_squared'], *dfs, averaging_100_loss['r_squared']])
    r_squared_metrics = df.groupby('Method')['R^2'].mean().to_dict()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=80)
    sns.boxplot(x='Task', y='R^2', hue='Method', data=df, ax=ax)
    #make x-labels rotated 
    plt.xticks(rotation=45)
    plt.ylim(-1, 1)
    #horizontal line at 0
    #ax.axhline(0, color='black', linestyle='--')
    # grid
    ax.grid(True)
    r_squared_string = ', '.join([f"{k}: {v:.3g}" for k, v in r_squared_metrics.items()])
    fig.suptitle(f"Epoch: {epoch:.1f}, {r_squared_string}")
    fig.tight_layout()
    return fig, ax

def load_model(path, model_name):
    # Load the configuration
    state_dict = torch.load(f"{path}Models/{model_name}.pt", weights_only=False)
    config = torch.load(f"{path}Models/{model_name}.config", weights_only=False)
    
    model_factory = {
        'context_size': config['context_size'] if 'context_size' in config else 256,
        'in_channels': config['in_channels'] if 'in_channels' in config else 30,
        'channels': config['channels'],
        'latent_dim': config['latent_dim'],
        'num_layers': config['num_layers'] if 'num_layers' in config else 4,
        'activation': config['activation'],
        'std_model': config['std_model'],
        'num_samples': config['num_samples'] if 'num_samples' in config else 585,
        'embedding_type': config['embedding_type'],
        'embedding_size': config['embedding_size'],
        'cosine_norm': config['cosine_norm'],
        'cosine_reverse_proj': config['cosine_reverse_proj'],
        'encoder_cond': config['encoder_cond'],
        'decoder_cond': config['decoder_cond'],
        'interpolated_residual': config['interpolated_residual']
    }
    
    if config['variational_model'] == 'ae':
        model = ERPAE(**model_factory)
    elif config['variational_model'] == 'unet':
        model = ERPUNet(**model_factory)
    elif config['variational_model'] == 'cslp-ae':
        model = CSLPAE(**model_factory)
    else:
        raise ValueError(f"Variational model {config['variational_model']} not recognized.")
    
    model.load_state_dict({n.replace("model.", ""): p for n, p in state_dict.items() if 'model.' in n})
    model.eval()
    return config, model


def get_grid_index(pca_min, pca_max, num_cells, pca_val):
    return tuple([int((pca_val[i] - pca_min[i])/(pca_max[i] - pca_min[i])*num_cells) for i in range(len(pca_val))])

def plot_trial_grid(fig, ax, results, dataset, test_subjects, test_task, num_cells=70, colorbar_aspect_ratio=10):
    result_trials = []
    result_latents = []
    for test_subject in test_subjects:
        num_samples = dataset.get_num_samples(test_subject, test_task, target=False)
        test_steps = list(range(1, num_samples))
        result_trials.extend(test_steps)
        for i in range(len(test_steps)):
            result_latents.append(results[test_subject][test_task][i])
    result_latents = torch.stack(result_latents).flatten(-2).cpu().numpy() #(num_samples-1, repeats, latent_dim)

    pca = PCA(n_components=2)
    result_latents_pca = pca.fit_transform(result_latents.reshape(-1, result_latents.shape[-1]))
    pca_min, pca_max = result_latents_pca.min(axis=0), result_latents_pca.max(axis=0)
    result_latents_pca = result_latents_pca.reshape(result_latents.shape[0], result_latents.shape[1], -1)

    # Create grid over PCA space and accumulate number of trials in each cell

    grid_dict = defaultdict(list)
    for k, trial in enumerate(result_trials):
        for i in range(result_latents_pca.shape[1]):
            grid_index = get_grid_index(pca_min, pca_max, num_cells-1, result_latents_pca[k, i])
            grid_dict[grid_index].append(trial)
    # Calculate mean number of trials in each cell
    grid_mean = {k: np.mean(v) for k, v in grid_dict.items()}

    # Create grid array
    grid_array = np.full((num_cells, num_cells), np.nan)
    for k, v in grid_mean.items():
        grid_array[k] = v
    # Plot grid and use grey color for cells without trials
    masked_array = np.ma.array(grid_array, mask=np.isnan(grid_array))
    cmap = plt.cm.jet
    cmap.set_bad('white',1.)
    im = ax.imshow(masked_array, interpolation='nearest', cmap=cmap)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, aspect=colorbar_aspect_ratio)
    # Set colorbar label
    cbar.set_label('Mean number of trials')
    ax.set_title(f"Task: {task_names[test_task]}")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    return fig, ax

def mle_distribution(ps):
    means = torch.stack([p.mean for p in ps], dim=1)
    variances = torch.stack([p.variance for p in ps], dim=1)
    precisions = 1 / variances

    weighted_mean = (means * precisions).sum(dim=1) / precisions.sum(dim=1)
    combined_variance = 1 / precisions.sum(dim=1)
    combined_std = combined_variance.sqrt()

    return torch.distributions.Normal(weighted_mean, combined_std)


def mean_amplitude(signals, start_idx, end_idx):
    return signals[:, start_idx:end_idx].mean(dim=1)


def peak_amplitude(signals, start_idx, end_idx, polarity="positive"):
    segment = signals[:, start_idx:end_idx]
    return segment.amax(dim=1) if polarity == "positive" else segment.amin(dim=1)


def peak_latency(signals, t, start_idx, end_idx, polarity="positive"):
    segment = signals[:, start_idx:end_idx]
    peak_idx = segment.argmax(dim=1) if polarity == "positive" else segment.argmin(dim=1)
    return t[start_idx:end_idx][peak_idx]


def area_latency(signals, t, start_idx, end_idx, fraction=0.5, polarity="positive"):
    segment = signals[:, start_idx:end_idx]
    time_segment = t[start_idx:end_idx]
    dt = time_segment[1] - time_segment[0]

    segment = torch.abs(segment)
    
    auc = torch.trapz(segment, dx=dt, dim=1)  # Area under the curve
    cumulative_auc = torch.cumsum(segment * dt, dim=1)  # Cumulative area
    
    target_idx = (cumulative_auc >= auc.unsqueeze(1) * fraction).max(dim=1).indices
    return time_segment[target_idx]

def onset_latency(signals, t, start_idx, end_idx, threshold, polarity="positive"):
    segment = signals[:, start_idx:end_idx]
    time_segment = t[start_idx:end_idx]
    
    onset = (segment >= threshold.unsqueeze(1)).max(dim=1) if polarity == "positive" else (segment <= threshold.unsqueeze(1)).max(dim=1)
    return time_segment[onset.indices]

time_window = {
    'P3': (0.300, 0.600),
    'N170': (0.110, 0.150),
    'MMN': (0.125, 0.225),
    'N400': (0.300, 0.500),
    'ERN': (0, 0.100),
    'N2pc': (0.200, 0.275),
    'LRP': (-0.100, 0),
}

epoch_window = defaultdict(lambda: (-0.2, 0.8))
epoch_window['LRP'] = (-0.8, 0.2)
epoch_window['ERN'] = (-0.6, 0.4)

polarity_name = {
    'P3': 'positive',
    'N170': 'negative',
    'MMN': 'negative',
    'N400': 'negative',
    'ERN': 'negative',
    'N2pc': 'negative',
    'LRP': 'negative',
}

def get_t(paradigm, fs=256, device='cuda'):
    epoch_start, epoch_end = epoch_window[paradigm]
    
    # Special handling for LRP and ERN
    if paradigm in ['LRP', 'ERN']:
        t = torch.linspace(epoch_start, epoch_end - 1 / fs, fs, device=device)
    else:
        t = torch.linspace(epoch_start + 1 / fs, epoch_end, fs, device=device)
    
    return t

def get_measures(x, paradigm, fs=256):
    t = get_t(paradigm, fs, device=x.device)

    w_start_t, w_end_t = time_window[paradigm]
    e_start_t, e_end_t = epoch_window[paradigm]

    polarity = polarity_name[paradigm]

    start_idx = int((w_start_t-e_start_t) * fs)
    end_idx = int((w_end_t-e_start_t) * fs)

    # Example calculations
    mean_amp = mean_amplitude(x, start_idx, end_idx)
    peak_amp = peak_amplitude(x, start_idx, end_idx, polarity=polarity)
    peak_lat = peak_latency(x, t, start_idx, end_idx, polarity=polarity)
    area_lat = area_latency(x, t, start_idx, end_idx, fraction=0.5, polarity=polarity)
    onset_lat = onset_latency(x, t, start_idx, end_idx, threshold=0.5 * peak_amp, polarity=polarity)

    return {
        'Mean Amp.': mean_amp,
        'Peak Amp.': peak_amp,
        'Peak Lat.': peak_lat,
        'Area Lat.': area_lat,
        'Onset Lat.': onset_lat
    }