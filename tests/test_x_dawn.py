#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch.set_float32_matmul_precision('high')

from data import ERPDataset, ERPBootstrapTargets, ERPCoreNormalizer, ModelState, Inputs, Targets, R2Metric
from util import get_measures

import pandas as pd


def evaluate_model(eval_predict_fn, metrics, test_dataset, test_bootstrap_targets, exclude_from_df=None):
    if exclude_from_df is None:
        exclude_from_df = set()
    with torch.inference_mode():
        steps = [5, 0.1, float('inf')]
        steps_str = ['K=5', '10%', '100%']
        test_dataset.deterministic()
        
        results = test_bootstrap_targets.get_results(eval_predict_fn, metrics, test_dataset, steps, tqdm_disabled=False)
        
        subjects = test_dataset.unique_subjects
        tasks = test_dataset.unique_tasks

        
        df = get_dataframe(results, subjects, tasks, test_dataset, 'XDAWN_Train', steps_str, exclude_from_df)
    return df, results

def get_dataframe(results, subjects, tasks, test_dataset, name, steps_str, exclude_from_df=None):
    if exclude_from_df is None:
        exclude_from_df = set()
    data = []
    metric_names = [n for n in results.keys() if n not in exclude_from_df]
    for s in subjects:
        for t in tasks:
            row = [name, s, test_dataset.task_to_label[t]]
            for metric_name in metric_names:
                for v in results[metric_name][s][t]:
                    if metric_name in ['MSE', 'True Var.', 'Model Var.', 'Pred. Var.']:
                        row.append(1e12*v)
                    elif metric_name in ['RMSE']:
                        row.append(1e6*v)
                    else:
                        row.append(v)
            data.append(row)
        
    columns = ['Model', 'Subject', 'Task']
    for metric_name in metric_names:
        for step in steps_str:
            columns.append(f"{metric_name} {step}")
    df = pd.DataFrame(data, columns=columns)
    return df

is_model = False
group = 'none'
model_config = {
    'sample_method': 'uniform',
    'input_data': 'single',
    'target_data': 'bootstrap_erp',
    'data_normalize': 'standard',
    'num_samples': 585,
    'cond_type': 'step',
    'group': 'XDAWN_TRAIN',
}
group = model_config['group']

print("Loading data...")
path = 'data/BCISpeller/'
split = 'test'
repeats = 200
test_dataset = ERPDataset(path=path, split=split, processing='simple', sample_method=model_config['sample_method'], input_data=model_config['input_data'], target_data=model_config['target_data'], normalize_kind=model_config['data_normalize'], num_samples=model_config['num_samples'], no_leakage=True, restricted=True).deterministic()
test_bootstrap_targets = ERPBootstrapTargets(path, split=split, processing='simple', repeats=repeats, input_type=model_config['cond_type'], prominent_channel_only=True)
normalizer = ERPCoreNormalizer(path=path, processing='simple', normalize_kind=model_config['data_normalize'])

def eval_predict_fn(x, step, s, t, channel):
    pred = ModelState(x=x)
    return pred

metrics = [Inputs(), Targets()]

df, results = evaluate_model(eval_predict_fn, metrics, test_dataset, test_bootstrap_targets, exclude_from_df={'Prediction'})
#%%
subjects = test_dataset.unique_subjects
tasks = test_dataset.unique_tasks
paradigms = {'P300': [1, 2]}
#%%
used_repeats = 15
channel = 31
from tqdm import tqdm
import numpy as np
import mne
from mne.preprocessing import Xdawn
from collections import defaultdict

xdawn_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for s in tqdm(subjects):
    for paradigm, (t1, t2) in paradigms.items():
        for v1, v2 in zip(results['Inputs'][s][t1], results['Inputs'][s][t2]):
            # Convert torch tensors to numpy arrays if needed.
            if torch.is_tensor(v1):
                v1_np = v1.detach().cpu().numpy()
            else:
                v1_np = v1
            if torch.is_tensor(v2):
                v2_np = v2.detach().cpu().numpy()
            else:
                v2_np = v2

            v1_repeats = []
            v2_repeats = []
            for i in range(used_repeats):
                # Determine the number of trials in each condition.
                n_trials_v1 = v1_np.shape[1]
                n_trials_v2 = v2_np.shape[1]
                
                # Use half the trials for training and the remaining for testing.
                train_count_v1 = n_trials_v1 // 2
                train_count_v2 = n_trials_v2 // 2

                # Split v1 and v2 into training and testing sets.
                # Here we assume that the trial axis is axis=0 after indexing by the repeat.
                v1_train = v1_np[i][:train_count_v1]
                v1_test  = v1_np[i][train_count_v1:]
                v2_train = v2_np[i][:train_count_v2]
                v2_test  = v2_np[i][train_count_v2:]
                
                # Create training data by concatenating the training halves.
                train_concat = np.concatenate([v1_train, v2_train], axis=0)

                # Build the MNE Info object.
                biosemi_montage = mne.channels.make_standard_montage('biosemi32')
                info = mne.create_info(ch_names=biosemi_montage.ch_names, sfreq=256., ch_types='eeg')

                # Build training events:
                total_train = train_concat.shape[0]
                events_train = np.zeros((total_train, 3), dtype=int)
                events_train[:, 0] = np.arange(total_train)  # dummy sample indices
                events_train[:train_count_v1, 2] = 1  # label for v1 training trials
                events_train[train_count_v1:, 2] = 2  # label for v2 training trials

                # Create training epochs.
                tmin = -0.2  # adjust as needed
                epochs_train = mne.EpochsArray(train_concat, info, events_train, tmin=tmin, verbose=False)

                # Train the xDAWN filter using the training epochs.
                n_components = 2  # adjust as needed
                xd = Xdawn(n_components=n_components, correct_overlap=False)
                xd.fit(epochs_train)

                # Prepare testing data by concatenating the testing halves.
                test_concat = np.concatenate([v1_test, v2_test], axis=0)
                n_test_v1 = v1_test.shape[0]
                n_test_v2 = v2_test.shape[0]
                total_test = n_test_v1 + n_test_v2

                # Build testing events.
                events_test = np.zeros((total_test, 3), dtype=int)
                events_test[:, 0] = np.arange(total_test)
                events_test[:n_test_v1, 2] = 1  # label for v1 testing trials
                events_test[n_test_v1:, 2] = 2  # label for v2 testing trials

                # Create testing epochs.
                epochs_test = mne.EpochsArray(test_concat, info, events_test, tmin=tmin, verbose=False)

                # Apply the trained xDAWN filter to the testing epochs.
                epochs_denoised = xd.apply(epochs_test)

                # Retrieve the denoised data for each condition.
                # Here we select channel index 31 and average across the test trials.
                v1_denoised = epochs_denoised['1'].get_data()[:, channel].mean(0)
                v2_denoised = epochs_denoised['2'].get_data()[:, channel].mean(0)

                # Store the denoised data for this repeat.
                v1_repeats.append(v1_denoised)
                v2_repeats.append(v2_denoised)

            # Stack the results from all repeats.
            v1_denoised_all = np.stack(v1_repeats, axis=0)
            v2_denoised_all = np.stack(v2_repeats, axis=0)

            # Save the outputs for the current subject and paradigm.
            xdawn_results['XDAWN'][s][t1].append(v1_denoised_all)
            xdawn_results['XDAWN'][s][t2].append(v2_denoised_all)
#%%
r2_metric = R2Metric()
r2_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for s in subjects:
    for t in tasks:
        for i in range(3): #number of steps
            x_hat = torch.from_numpy(xdawn_results['XDAWN'][s][t][i])
            target = results['Targets'][s][t][i]
            target = target.expand_as(x_hat)
            tss = torch.sum(target.square())
            rss = torch.sum((target - x_hat).square())
            r2_results['R2'][s][t].append((1 - rss / tss).item())
#%%
# Create dataframe for xDAWN results
xdawn_df = get_dataframe(r2_results, subjects, tasks, test_dataset, 'XDAWN', ['K=5', '10%', '100%'], exclude_from_df={'Inputs', 'Targets'})
# Set index to Model, Subject, Task
xdawn_df.set_index(['Model', 'Subject', 'Task'], inplace=True)
# %%
# Show the different groupby means
print("Mean over subjects")
xdawn_grouped = xdawn_df.groupby(['Model', 'Task']).mean()
print(xdawn_grouped)
# %%
# Show the different groupby means
print("Mean over tasks")
xdawn_grouped = xdawn_df.groupby(['Model', 'Subject']).mean()
print(xdawn_grouped)
# %%
# Show the full mean
print("Mean over all")
xdawn_grouped = xdawn_df.groupby(['Model']).mean()
print(xdawn_grouped)
# %%
# Mean over subjects and print with +-sem as well. Pretty print as markdown table
xdawn_grouped = (xdawn_df * 100).groupby(['Model', 'Task']).agg(['mean', 'sem'])
# Print the mean and sem as a markdown table
xdawn_grouped = xdawn_grouped.reset_index()
#%%
xdawn_grouped['Mean'] = xdawn_grouped['Mean'].apply(lambda x: f"{x*100:.2f}")
xdawn_grouped['SEM'] = xdawn_grouped['SEM'].apply(lambda x: f"{x*100:.2f}")
# %%
