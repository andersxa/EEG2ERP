#%%
import numpy as np
import torch
from eeg2erp.data import ERPDataset
from tqdm import tqdm
import math
#%%
# dataset = ERPDataset(path='data/BCISpeller/', split='train', processing='simple', no_leakage=False)
# max_trials = 0
# for s in dataset.unique_subjects:
#     for t in dataset.unique_tasks:
#         num_trials = len(dataset.get_indices(s, t))
#         if num_trials > max_trials:
#             max_trials = num_trials
# print(max_trials)
#%%
dataset = ERPDataset(path='data/BCISpeller/', split='test', processing='simple', num_samples=1500, no_leakage=True, restricted=True)

single_erp_target = True

REPEATS = 200
dataset.deterministic()
subjects = dataset.unique_subjects
tasks = dataset.unique_tasks
# For each pair of s and t, split the full indices into 2 equal parts (randomly)
half1_indices = dataset.half_targets
half2_indices = dataset.half_inputs
#%%

data = {}
with torch.inference_mode():
    s_t_target = {s: {t: [] for t in tasks} for s in subjects}
    s_t_bootstrap_indices = {s: {t: None for t in tasks} for s in subjects}
    s_t_steps = {s: {t: None for t in tasks} for s in subjects}
    total = len(subjects)*len(tasks)*REPEATS
    t_bar = tqdm(total=total)
    for s in subjects:
        for t in tasks:
            for r in range(REPEATS):
                # Sample targets from half1
                if not single_erp_target:
                    target_indices = dataset.rng.choice(half1_indices[s][t], len(half1_indices[s][t]))
                    target_data = dataset.data[target_indices].mean(dim=0)
                    s_t_target[s][t].append(target_data)
                else:
                    s_t_target[s][t] = dataset.data[half1_indices[s][t]].mean(dim=0)
                
                # Create all possible bootstraps
                num_trials = len(half2_indices[s][t])
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
                if s_t_bootstrap_indices[s][t] is None:
                    s_t_bootstrap_indices[s][t] = {j: [] for j in steps}
                
                # For each possible bootstrap size (up to num_trials)
                for j in steps:
                    bootstrap_indices = dataset.rng.choice(half2_indices[s][t], j+1)
                    s_t_bootstrap_indices[s][t][j].append(bootstrap_indices)
                t_bar.update(1)
            if not single_erp_target:
                s_t_target[s][t] = torch.stack(s_t_target[s][t], dim=0)
            else:
                s_t_target[s][t] = s_t_target[s][t].unsqueeze(0)
            for j in s_t_bootstrap_indices[s][t].keys():
                s_t_bootstrap_indices[s][t][j] = np.stack(s_t_bootstrap_indices[s][t][j], axis=0)
data = {
    'target': s_t_target,
    'bootstrap_indices': s_t_bootstrap_indices,
    'steps': s_t_steps
}
#%%
torch.save(data, f"data/BCISpeller/{dataset.processing}_{dataset.split}_{REPEATS}_half_half_bootstrap.pt")
#%%