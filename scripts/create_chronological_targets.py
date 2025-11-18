#%%
import numpy as np
import torch
from collections import defaultdict
from eeg2erp.data import ERPDataset
from tqdm import tqdm
import math
#%%
dataset = ERPDataset(path='data/', split='test', processing='simple', num_samples=585, no_leakage=True, restricted=True)

single_erp_target = True

# No repeats needed for chronological sampling - we just take the first K trials
dataset.deterministic()
subjects = dataset.unique_subjects
tasks = dataset.unique_tasks

# For chronological sampling, we use the full chronological indices instead of random halves
# But we still need targets from one half and inputs from the other for no-leakage
half1_indices = dataset.half_targets  # Use for targets
half2_indices = dataset.half_inputs   # Use for chronological sampling
#%%

data = {}
with torch.inference_mode():
    s_t_target = {s: {t: [] for t in tasks} for s in subjects}
    s_t_bootstrap_indices = {s: {t: None for t in tasks} for s in subjects}
    s_t_steps = {s: {t: None for t in tasks} for s in subjects}
    total = len(subjects)*len(tasks)
    t_bar = tqdm(total=total)

    for s in subjects:
        for t in tasks:
            # Single target from half1 (same as original)
            if single_erp_target:
                s_t_target[s][t] = dataset.data[half1_indices[s][t]].mean(dim=0).unsqueeze(0)

            # Get chronological indices from half2 (these are already in chronological order)
            chronological_indices = half2_indices[s][t]
            num_trials = len(chronological_indices)

            # Create steps (same logic as original)
            steps = None
            if num_trials < 50 or dataset.split == 'test':
                steps = list(range(num_trials+1))
            else:
                # If there are more than 50 trials, we will start with 50 and then do 5% increments until we reach 100%
                steps = list(range(50+1))
                # 50/num_trials -> round to nearest multiple of 5% -> multiply by num_trials
                nearest_multiple = math.ceil(50/num_trials*20)/20
                start_after = int(nearest_multiple*num_trials)
                step_size = int(num_trials*0.05)
                steps += list(range(start_after, num_trials, step_size))
                if num_trials not in steps:
                    steps.append(num_trials)

            s_t_steps[s][t] = steps
            s_t_bootstrap_indices[s][t] = {}

            # For each possible bootstrap size (up to num_trials)
            # Instead of random sampling with repeats, we take the first j+1 chronological indices
            for j in steps:
                if j == 0:
                    # For j=0, we need at least 1 trial
                    bootstrap_indices = np.array([chronological_indices[0]])
                else:
                    # Take the first j+1 chronological trials
                    num_to_take = min(j+1, len(chronological_indices))
                    bootstrap_indices = np.array(chronological_indices[:num_to_take])

                # Add an extra dimension to match the expected format (1, num_trials)
                # This way we don't need to modify the data.py methods
                s_t_bootstrap_indices[s][t][j] = bootstrap_indices[np.newaxis, :]

            t_bar.update(1)

data = {
    'target': s_t_target,
    'bootstrap_indices': s_t_bootstrap_indices,
    'steps': s_t_steps
}
#%%
torch.save(data, f"data/{dataset.processing}_{dataset.split}_chronological_bootstrap.pt")
print(f"Saved chronological bootstrap targets to: data/{dataset.processing}_{dataset.split}_chronological_bootstrap.pt")
#%%