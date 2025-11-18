#%%
from collections import defaultdict
from eeg2erp.data import ERPDataset
import torch
path = 'data/'
processing = 'simple'
dataset = ERPDataset(path, split='train', processing=processing)
template_dict = defaultdict(list)
for t in dataset.unique_tasks:
    for s in dataset.unique_subjects:
        indices = (dataset.tasks == t) & (dataset.subjects == s)
        data = dataset.data[indices][:, dataset.task_to_channel[t]]
        mean = data.mean(dim=0)
        template_dict[t].append(mean)
    template_dict[t] = torch.stack(template_dict[t])
template_dict = dict(template_dict)
torch.save(template_dict, f"{path}/{processing}_{dataset.split}_neighbors.pt")
#%%