#%%
import torch

pt1 = torch.load('data/simple_data.pt', weights_only=False)
#%%
unique_subjects = pt1['subjects'].unique().numpy().tolist()
# Split unique subjects into 70% train, 20% test, 10% val
n = len(unique_subjects)
n_train = int(0.7 * n)
n_test = int(0.2 * n)
n_val = n - n_train - n_test
# Shuffle the list first
import random
random.seed(0)
random.shuffle(unique_subjects)
print(list(sorted(unique_subjects[:n_train])))
print(list(sorted(unique_subjects[n_train:n_train+n_test])))
print(list(sorted(unique_subjects[n_train+n_test:])))
# %%
# Plot the trials
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
i = 101
trial_1 = pt1['data'][pt1['tasks'] == 1].mean(dim=0)[i]
trial_2 = pt1['data'][pt1['tasks'] == 2].mean(dim=0)[i]
trial_3 = pt1['data'][pt1['tasks'] == 3].mean(dim=0)[i]
ax.plot(trial_1.numpy(), label='Trial 1')
ax.plot(trial_2.numpy(), label='Trial 2')
ax.plot(trial_3.numpy(), label='Trial 3')
plt.title(f'Channel {i}')
plt.show()
#%%
from data import ERPDataset

path = 'dataM/'
processing = 'simple'
dataset = ERPDataset(path, split='train', processing=processing)
dataset.save_mean_std()
# %%
# For each subject and task pair, get the max number of trials available
max_num_trials = 0
min_num_trials = float('inf')
for s in dataset.unique_subjects:
    for t in dataset.unique_tasks:
        num_trials = dataset.get_num_samples(s, t, target=True)
        if num_trials > max_num_trials:
            max_num_trials = num_trials
        if num_trials < min_num_trials:
            min_num_trials = num_trials
print(min_num_trials, max_num_trials)
# %%
