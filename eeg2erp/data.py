import torch
import torch.distributions as td
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
from dataclasses import dataclass
import math

class ERPTargets():
    def __init__(self, path, processing='simple', target_type='subject'):
        self.targets = torch.load(f"{path}{processing}_{target_type}_targets.pt", weights_only=False)
        self.size = len(self.targets)
    def get_targets(self, subjects=None, tasks=None):
        if tasks is None:
            raise ValueError("Tasks must be specified.")
        if subjects is None:
            return torch.stack([self.targets[t] for t in tasks], dim=0)
        else:
            return torch.stack([self.targets[s][t] for s, t in zip(subjects, tasks)], dim=0)

def closest_val(val, iterable):
    max_step = max(k for k in iterable)
    if isinstance(val, float):
        if val > 1:
            val = max_step
        elif val < 0:
            raise ValueError("Negative values not allowed.")
        else:
            val = int(val * max_step)
    closest_step = next((k for k in iterable if k >= val), max_step)
    return closest_step

class ERPBootstrapTargets():
    def __init__(self, path, split='test', processing='simple', repeats=32, input_type='step', prominent_channel_only=False, chronological_sampling=False):
        if chronological_sampling:
            # Load chronological targets (no repeats needed)
            data = torch.load(f"{path}{processing}_{split}_chronological_bootstrap.pt", weights_only=False)
            self.repeats = 1  # Only one "repeat" for chronological data
        else:
            # Load regular bootstrap targets with repeats
            data = torch.load(f"{path}{processing}_{split}_{repeats}_half_half_bootstrap.pt", weights_only=False)
            self.repeats = repeats

        self.targets = data['target']
        self.bootstrap_indices = data['bootstrap_indices']
        self.steps = data['steps']
        self.input_type = input_type
        self.prominent_channel_only = prominent_channel_only
        self.split = split
        self.chronological_sampling = chronological_sampling

    def reconfigure(self, input_type=None, chronological_sampling=None):
        if input_type is not None:
            self.input_type = input_type
        if chronological_sampling is not None:
            self.chronological_sampling = chronological_sampling
        return self

    def to(self, device):
        for s in self.targets.keys():
            for t in self.targets[s].keys():
                self.targets[s][t] = self.targets[s][t].to(device)
        return self

    @torch.no_grad()
    def get_loss(self, predict_fn, loss_fn, dataset, subjects=None, tasks=None, single_in=False, channel_in=False, tqdm_disabled=False, r_squared_step=5):
        if subjects is None:
            subjects = dataset.unique_subjects
        if tasks is None:
            tasks = dataset.unique_tasks
        loss_dict = {s: {t: [] for t in tasks} for s in subjects}
        r_squared_dict = {s: {t: 0.0 for t in tasks} for s in subjects}
        predictions = {s: {t: None for t in tasks} for s in subjects}
        total_loss = 0.0
        with tqdm(total=len(subjects)*len(tasks), disable=tqdm_disabled) as t_bar:
            for s in subjects:
                for t in tasks:
                    target = self.targets[s][t]
                    if self.split == 'test':
                        target = target.unsqueeze(0)
                    if self.prominent_channel_only or channel_in:
                        target = target[:, dataset.task_to_channel[t]]
                    bootstrap_index = self.bootstrap_indices[s][t]
                    max_step = max(k for k in bootstrap_index.keys())
                    # the first element that is >= r_squared_step
                    closest_step = next((k for k in bootstrap_index.keys() if k >= r_squared_step), max_step)
                    for j in bootstrap_index.keys():
                        bootstrap = dataset.data[bootstrap_index[j]]
                        if channel_in:
                            bootstrap = bootstrap[:, :, dataset.task_to_channel[t]]
                        if not single_in:
                            bootstrap = bootstrap.mean(dim=1)
                        pred = predict_fn(bootstrap, j, s, t)
                        if self.prominent_channel_only and not channel_in:
                            pred = pred[:, dataset.task_to_channel[t]]
                        loss = loss_fn(pred, target)
                        loss_dict[s][t].append(loss)
                        if j == closest_step:
                            predictions[s][t] = (pred.mean(dim=0).cpu().numpy(), pred.std(dim=0).cpu().numpy() / math.sqrt(pred.size(0)))
                            tss = torch.sum(target.square())
                            rss = torch.sum((target - pred).square())
                            r_squared_dict[s][t] = (1 - rss / tss).item()
                    loss_dict[s][t] = torch.stack(loss_dict[s][t], dim=0).cpu().numpy()
                    cur_loss = loss_dict[s][t].mean()
                    total_loss += cur_loss
                    if not tqdm_disabled:
                        t_bar.set_postfix_str(f"Subject {s}, Task {t}, Loss: {1e12*cur_loss / (len(subjects)*len(tasks)):.3g}, Total Loss: {1e12*total_loss / (len(subjects)*len(tasks)):.3g}, R^2: {r_squared_dict[s][t]:.3g}")
                        t_bar.update(1)
        total_loss = 1e12*total_loss / (len(subjects)*len(tasks))
        return loss_dict, r_squared_dict, predictions, total_loss
    
    @torch.no_grad()
    def get_r_squared(self, predict_fn, dataset, subjects=None, tasks=None, single_in=False, channel_in=False, tqdm_disabled=False, r_squared_step=5):
        if subjects is None:
            subjects = dataset.unique_subjects
        if tasks is None:
            tasks = dataset.unique_tasks
        r_squared_dict = {s: {t: 0.0 for t in tasks} for s in subjects}
        with tqdm(total=len(subjects)*len(tasks), disable=tqdm_disabled) as t_bar:
            for s in subjects:
                for t in tasks:
                    target = self.targets[s][t]
                    if self.prominent_channel_only or channel_in:
                        target = target[:, dataset.task_to_channel[t]]
                    bootstrap_index = self.bootstrap_indices[s][t]
                    max_step = max(k for k in bootstrap_index.keys())
                    # the first element that is >= r_squared_step
                    closest_step = next((k for k in bootstrap_index.keys() if k >= r_squared_step), max_step)
                    bootstrap = dataset.data[bootstrap_index[closest_step]]
                    if channel_in:
                        bootstrap = bootstrap[:, :, dataset.task_to_channel[t]]
                    if not single_in:
                        bootstrap = bootstrap.mean(dim=1)
                    pred = predict_fn(bootstrap, closest_step, s, t)
                    if self.prominent_channel_only and not channel_in:
                        pred = pred[:, dataset.task_to_channel[t]]
                    tss = torch.sum(target.square())
                    rss = torch.sum((target - pred).square())
                    r_squared_dict[s][t] = (1 - rss / tss).item()
                    if not tqdm_disabled:
                        t_bar.set_postfix_str(f"Subject {s}, Task {t}, R^2: {r_squared_dict[s][t]:.3g}")
                        t_bar.update(1)
        return r_squared_dict
    

    @torch.no_grad()
    def get_results(self, predict_fn, metrics, dataset, step=None, subjects=None, tasks=None, tqdm_disabled=False, only_index=None):
        if subjects is None:
            subjects = dataset.unique_subjects
        if tasks is None:
            tasks = dataset.unique_tasks
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        with tqdm(total=len(subjects)*len(tasks), disable=tqdm_disabled) as t_bar:
            for s in subjects:
                for t in tasks:
                    target = self.targets[s][t]
                    bootstrap_index = self.bootstrap_indices[s][t]
                    channel = dataset.task_to_channel[t]

                    indices_to_process = list(bootstrap_index.keys())
                    steps = step[s][t] if isinstance(step, dict) else step
                    if isinstance(steps, int) or isinstance(steps, float):
                        indices_to_process = [closest_val(steps, bootstrap_index.keys())]
                    elif steps is not None:
                        indices_to_process = []
                        for k in steps:
                            indices_to_process.append(closest_val(k, bootstrap_index.keys()))

                    for j, k in enumerate(indices_to_process):
                        bootstrap = dataset.data[bootstrap_index[k]]
                        if only_index is not None:
                            bootstrap = bootstrap[only_index:only_index+1]
                        pred = predict_fn(bootstrap, k, s, t, channel)
                        for metric in metrics:
                            results[metric.name][s][t].append(metric.calc(pred, target, channel))
                        results['Steps'][s][t].append((j, k))
                    t_bar.update(1)
        # Convert results to dictionaries
        results = dict(results)
        names = [metric.name for metric in metrics] + ['Steps']
        for name in names:
            results[name] = dict(results[name])
            for s in results[name].keys():
                results[name][s] = dict(results[name][s])
        return results

@dataclass
class ModelState():
    x: torch.Tensor = None
    x_hat: torch.Tensor = None
    z: torch.Tensor = None
    p: td.Normal = None
    q: td.Normal = None
    nfe: int = 0
    picked: bool = False

    def pick_channel(self, channel):
        if not self.picked:
            if self.x is not None:
                self.x = self.x[..., channel, :]
            if self.x_hat is not None:
                self.x_hat = self.x_hat[..., channel, :]
            if self.p is not None:
                self.p = td.Normal(self.p.mean[..., channel, :], self.p.scale[..., channel, :])
            self.picked = True
        return self

class DefaultMetric():
    def __init__(self, name, single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        raise NotImplementedError("calc_fn not implemented.")

    def calc(self, pred: ModelState, target: torch.Tensor, channel: int):
        if self.single_channel:
            pred = pred.pick_channel(channel)
        target = target[..., channel, :]
        return self.calc_fn(pred, target)

class MSEMetric(DefaultMetric):
    def __init__(self, name="MSE", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return torch.mean((pred.x_hat - target).square()).item()

class GaussianNLLMetric(DefaultMetric):
    def __init__(self, name="NLL", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        target = target.expand_as(pred.p.mean)
        return -torch.mean(pred.p.log_prob(target)).item()

class R2Metric(DefaultMetric):
    def __init__(self, name="R^2", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        target = target.expand_as(pred.x_hat)
        tss = torch.sum(target.square())
        rss = torch.sum((target - pred.x_hat).square())
        return (1 - rss / tss).item()

class R2WeightedMetric(DefaultMetric):
    def __init__(self, name="WR^2", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        mode, std = pred.p.mean, pred.p.scale
        std_invsq = std**(-2)
        target = target.expand_as(mode)
        rss = torch.sum(std_invsq*(target - mode).square())
        tss = torch.sum(std_invsq*target.square())
        return (1 - rss / tss).item()

class NFEMetric(DefaultMetric):
    def __init__(self, name="NFE", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return pred.nfe

class TrueVariance(DefaultMetric):
    def __init__(self, name="True Var.", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return torch.var(target).item()

class FullTrueVariance(DefaultMetric):
    def __init__(self, name="Full True Var.", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return torch.var(target, dim=0)

class PredictionVariance(DefaultMetric):
    def __init__(self, name="Pred. Var.", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return torch.var(pred.x_hat, dim=1).mean().item()
    
class FullPredictionVariance(DefaultMetric):
    def __init__(self, name="Full Pred. Var.", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return torch.var(pred.x_hat, dim=1)
    
class FullInputVariance(DefaultMetric):
    def __init__(self, name="Full Input. Var.", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return torch.var(pred.x, dim=1)

class ModelVariance(DefaultMetric):
    def __init__(self, name="Model Var.", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return pred.p.variance.mean().item()
    
class ModelFullVariance(DefaultMetric):
    def __init__(self, name="Full Model Var.", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return pred.p.variance

class ModelPredictions(DefaultMetric):
    def __init__(self, name="Prediction", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return pred.x_hat

class ModelLatents(DefaultMetric):
    def __init__(self, name="Latents", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return pred.z

class Inputs(DefaultMetric):
    def __init__(self, name="Inputs", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return pred.x
    
class Targets(DefaultMetric):
    def __init__(self, name="Targets", single_channel=False):
        self.name = name
        self.single_channel = single_channel

    def calc_fn(self, pred: ModelState, target: torch.Tensor):
        return target


@dataclass
class ERPSample:
    x: torch.Tensor
    targets: torch.Tensor
    x_perc: torch.Tensor
    target_perc: torch.Tensor
    x_trials: torch.Tensor
    target_trials: torch.Tensor

    def to(self, device):
        if self.x is not None:
            if isinstance(self.x, td.Normal):
                self.x = td.Normal(self.x.mean.to(device), self.x.scale.to(device))
            else:
                self.x = self.x.to(device)
        if self.targets is not None:
            if isinstance(self.targets, td.Normal):
                self.targets = td.Normal(self.targets.mean.to(device), self.targets.scale.to(device))
            else:
                self.targets = self.targets.to(device)
        if self.x_perc is not None:
            self.x_perc = self.x_perc.to(device)
        if self.target_perc is not None:
            self.target_perc = self.target_perc.to(device)
        if self.x_trials is not None:
            self.x_trials = self.x_trials.to(device)
        if self.target_trials is not None:
            self.target_trials = self.target_trials.to(device)
        return self

class ERPCoreNormalizer():
    def __init__(self, path, processing='simple', normalize_kind='standard'):
        self.path = path
        self.processing = processing
        self.normalize_kind = normalize_kind
        self.mean = torch.load(f"{path}{processing}_train_mean.pt", weights_only=False)
        self.std = torch.load(f"{path}{processing}_train_std.pt", weights_only=False)

    def reconfigure(self, normalize_kind=None):
        if normalize_kind is not None:
            self.normalize_kind = normalize_kind
        return self

    def normalize(self, x):
        if self.normalize_kind == 'standard':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)
        elif self.normalize_kind == 'scale':
            return x / self.std.to(x.device)
        else:
            return x

    def denormalize(self, x):
        if self.normalize_kind == 'standard':
            return x * self.std.to(x.device) + self.mean.to(x.device)
        elif self.normalize_kind == 'scale':
            return x * self.std.to(x.device)
        else:
            return x

    # In case x is a torch.distributions.Normal object
    def normalize_dist(self, x):
        if self.normalize_kind == 'standard':
            return td.Normal((x.mean - self.mean.to(x.mean.device)) / self.std.to(x.mean.device), x.scale / self.std.to(x.mean.device))
        elif self.normalize_kind == 'scale':
            return td.Normal(x.mean / self.std.to(x.mean.device), x.scale / self.std.to(x.mean.device))
        else:
            return x

    # In case x is a torch.distributions.Normal object
    def denormalize_dist(self, x):
        if self.normalize_kind == 'standard':
            return td.Normal(x.mean * self.std.to(x.mean.device) + self.mean.to(x.mean.device), x.scale * self.std.to(x.mean.device))
        elif self.normalize_kind == 'scale':
            return td.Normal(x.mean * self.std.to(x.mean.device), x.scale * self.std.to(x.mean.device))
        else:
            return x

    def normalize_sample(self, sample):
        if sample.x is not None:
            if isinstance(sample.x, td.Normal):
                sample.x = self.normalize_dist(sample.x)
            else:
                sample.x = self.normalize(sample.x)
        return sample

    def denormalize_sample(self, sample):
        if sample.x is not None:
            if isinstance(sample.x, td.Normal):
                sample.x = self.denormalize_dist(sample.x)
            else:
                sample.x = self.denormalize(sample.x)
        return sample

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class ERPDataset():
    def __init__(self, path, split='train', processing='simple', sample_method='uniform', input_data='single', target_data='bootstrap_erp', normalize_kind='standard', num_samples=256, no_leakage=False, restricted=False, exclude_tasks=[], num_classes=4):
        self.path = path
        self.split = split
        self.processing = processing
        self.sample_method = sample_method
        self.input_data = input_data
        self.target_data = target_data
        self.normalize_kind = normalize_kind
        data_dict = torch.load(f"{path}{processing}_data.pt", weights_only=False)
        self.data = data_dict["data"]
        self.subjects = data_dict["subjects"].numpy()
        self.tasks = data_dict["tasks"].numpy()
        self.task_to_label = data_dict['labels']
        self.metadata = None
        if 'metadata' in data_dict:
            self.metadata = data_dict['metadata'].copy()
        self.num_samples = num_samples
        self.input_data = input_data
        self.target_data = target_data
        self.restricted = restricted

        if self.sample_method == 'weighted':
            weight_data = torch.load(f"{path}{processing}_num_trials_weights.pt", weights_only=False)
            s_t_weights = weight_data['s_t_weights']
            self.trial_weights = s_t_weights.sum(0) #index by task
            self.weight_task_to_task_index = weight_data['weight_task_to_task_index']

        if os.path.exists(f"{path}{processing}_train_mean.pt") and os.path.exists(f'{path}{processing}_train_std.pt'):
            self.mean = torch.load(f"{path}{processing}_train_mean.pt", weights_only=False)
            self.std = torch.load(f"{path}{processing}_train_std.pt", weights_only=False)
        else:
            print("Warning: Mean and std not found. You should run save_mean_std on training data to save the mean and std.")

        self.rng = np.random.default_rng(6078994882329848295) #reset the rng
        if 'dataset' in data_dict:
            if data_dict['dataset'].lower() == 'eegmmi':
                self.prepare_eegmmi(data_dict['train_subjects'], data_dict['test_subjects'], data_dict['dev_subjects'], num_classes=num_classes)
            elif data_dict['dataset'].lower() == 'wakemanhenson-eeg':
                self.prepare_WMH()
            elif data_dict['dataset'].lower() == 'wakemanhenson-meg':
                self.prepare_WMHM()
            elif data_dict['dataset'].lower() == 'bcispeller':
                self.prepare_BCISpeller(data_dict['train_subjects'], data_dict['test_subjects'], data_dict['dev_subjects'])
        else:
            self.prepare_erp_core()

        if split == 'dev':
            self.unique_subjects = self.dev_splits
        elif split == 'test':
            self.unique_subjects = self.test_splits
        elif split == 'train':
            self.unique_subjects = self.train_splits


        self.unique_tasks = [t for t, l in self.task_to_label.items() if t not in exclude_tasks]

        self.subject_indices = {s: [] for s in self.unique_subjects}
        self.task_indices = {t: [] for t in self.unique_tasks}

        data_indices = []
        for i, (s, t) in enumerate(zip(self.subjects, self.tasks)):
            if s not in self.subject_indices:
                continue
            if t not in self.task_indices:
                continue
            data_indices.append(i)
        self.data = self.data[data_indices].float().contiguous()
        self.subjects = np.ascontiguousarray(self.subjects[data_indices])
        self.tasks = np.ascontiguousarray(self.tasks[data_indices])
        self.size = len(self.data)


        self.full_indices = defaultdict(lambda: defaultdict(list))
        for i, (s, t) in enumerate(zip(self.subjects, self.tasks)):
            self.subject_indices[s].append(i)
            self.task_indices[t].append(i)
            self.full_indices[s][t].append(i)

        self.device = 'cpu'

        self.num_subjects = len(self.unique_subjects)
        self.num_tasks = len(self.unique_tasks)

        self.rng = np.random.default_rng(6078994882329848295) #reset the rng

        self.no_leakage = no_leakage
        if self.no_leakage:
            self.half_inputs = {s: {t: [] for t in self.unique_tasks} for s in self.unique_subjects}
            self.half_targets = {s: {t: [] for t in self.unique_tasks} for s in self.unique_subjects}
            for s in self.unique_subjects:
                for t in self.unique_tasks:
                    indices = self.full_indices[s][t].copy()
                    self.rng.shuffle(indices)
                    half = len(indices)//2
                    self.half_inputs[s][t] = indices[:half]
                    self.half_targets[s][t] = indices[half:]

            # Reset the rng to the original seed
            self.rng = np.random.default_rng(6078994882329848295)
            
    def save_mean_std(self):
        self.mean = self.data.mean(dim=-1, keepdim=True).mean(0, keepdim=True).float()
        self.std = self.data.std(dim=-1, keepdim=True).mean(0, keepdim=True).float()
        torch.save(self.mean, f"{self.path}{self.processing}_train_mean.pt")
        torch.save(self.std, f"{self.path}{self.processing}_train_std.pt")

    def prepare_erp_core(self):
        self.dev_splits = [4, 7, 27, 33]
        self.test_splits = [5, 14, 15, 20, 22, 23, 26, 29]
        self.train_splits = [1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 21, 24, 25, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40]

        self.task_to_channel = {
            0: 19, #FCz
            1: 19, #FCz
            2: 19, #FCz
            3: 19, #FCz
            4: 19, #FCz
            5: 19, #FCz
            6: 25, #PO8
            7: 25, #PO8
            8: 25, #PO8
            9: 25, #PO8
            10: 13, #CPz
            11: 13, #CPz
            12: 12, #Pz
            13: 12 #Pz
        }

        self.paradigm_components = {
            'ERN': [0, 1],
            'LRP': [2, 3],
            'MMN': [4, 5],
            'N2pc': [6, 7],
            'N170': [8, 9],
            'N400': [10, 11],
            'P3': [12, 13]
        }

    def prepare_eegmmi(self, train_subjects, test_subjects, dev_subjects, num_classes=4):
        self.train_splits = train_subjects
        self.dev_splits = dev_subjects
        self.test_splits = test_subjects
        # Given num_classes there are 4 possibilities: L/R, L/R/0, L/R/0/F, ALL
        # For all other than ALL, L is 0, R is 1, 0 is 2, F is 3
        self.class_convert_2 = {0: -1, 1: 0, 2: 1, 3: -1, 4: -1, 5: 0, 6: 1, 7: -1, 8: -1}
        self.class_convert_3 = {0: 2, 1: 0, 2: 1, 3: -1, 4: -1, 5: 0, 6: 1, 7: -1, 8: -1}
        self.class_convert_4 = {0: 2, 1: 0, 2: 1, 3: -1, 4: 3, 5: 0, 6: 1, 7: -1, 8: 3}
        if num_classes == 2:
            self.tasks = np.vectorize(self.class_convert_2.get)(self.tasks)
            self.task_to_label = {0: 'L', 1: 'R'}
        elif num_classes == 3:
            self.tasks = np.vectorize(self.class_convert_3.get)(self.tasks)
            self.task_to_label = {0: 'L', 1: 'R', 2: '0'}
        elif num_classes == 4:
            self.tasks = np.vectorize(self.class_convert_4.get)(self.tasks)
            self.task_to_label = {0: 'L', 1: 'R', 2: '0', 3: 'F'}
        else:
            self.tasks = np.vectorize(int)(self.tasks)
            self.task_to_label = {int(k): str(v) for k, v in self.task_to_label.items()}
    
    def prepare_WMH(self):
        self.train_splits = [1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14]
        self.dev_splits = [12, 15]
        self.test_splits = [0, 4, 6]
        
        self.task_to_channel = {
            1: 65,
            2: 65,
            3: 65,
        }
        
    def prepare_WMHM(self):
        self.train_splits = [1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14]
        self.dev_splits = [12, 15]
        self.test_splits = [0, 4, 6]
        
        self.task_to_channel = {
            1: 98,
            2: 98,
            3: 98,
        }
    
    def prepare_BCISpeller(self, train_subjects, test_subjects, dev_subjects):
        self.train_splits = train_subjects
        self.dev_splits = dev_subjects
        self.test_splits = test_subjects
        self.task_to_channel = {
            1: 31,
            2: 31,
        }

    def reconfigure(self, sample_method=None, input_data=None, target_data=None, normalize_kind=None, num_samples=None):
        if sample_method is not None:
            self.sample_method = sample_method
            if self.sample_method == 'weighted':
                weight_data = torch.load(f"{self.path}{self.processing}_num_trials_weights.pt", weights_only=False)
                s_t_weights = weight_data['s_t_weights']
                self.trial_weights = s_t_weights.sum(0) #index by task
                self.weight_task_to_task_index = weight_data['weight_task_to_task_index']
        if input_data is not None:
            self.input_data = input_data
        if target_data is not None:
            self.target_data = target_data
        if normalize_kind is not None:
            self.normalize_kind = normalize_kind
        if num_samples is not None:
            self.num_samples = num_samples
        return self

    def get_indices(self, s, t, target=False, use_full=False):
        if not self.no_leakage or use_full:
            return self.full_indices[s][t]
        elif target:
            return self.half_targets[s][t]
        else:
            return self.half_inputs[s][t]

    def get_num_samples(self, s, t, target=False):
        return len(self.get_indices(s, t, target)) if self.restricted else self.num_samples

    def __len__(self):
        return self.size

    def to(self, device, data=True):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        if data:
            self.data = self.data.to(device)
        self.device = device
        return self

    def deterministic(self, seed=6078994882329848295):
        self.rng = np.random.default_rng(seed)
        return self

    def normalize(self, x):
        if self.normalize_kind == 'standard':
            return (x - self.mean) / self.std
        elif self.normalize_kind == 'scale':
            return x / self.std
        else:
            return x

    def denormalize(self, x):
        if self.normalize_kind == 'standard':
            return x * self.std + self.mean
        elif self.normalize_kind == 'scale':
            return x * self.std
        else:
            return x

    def sample_batch(self, batch_size):
        subjects = self.rng.choice(self.unique_subjects, batch_size)
        tasks = self.rng.choice(self.unique_tasks, batch_size)
        return subjects, tasks, (batch_size,)

    def sample_pairwise(self, batch_size):
        subjects = self.rng.choice(self.unique_subjects, batch_size//2)
        tasks = self.rng.choice(self.unique_tasks, batch_size//2)
        subjects = np.concatenate([subjects, subjects])
        tasks = np.concatenate([tasks, tasks])
        return subjects, tasks, (2, batch_size//2,)

    def sample_contrastive(self, batch_size):
        one_round_num = len(self.unique_subjects)*len(self.unique_tasks) # S x T
        num_rounds = batch_size // one_round_num
        s, t = np.meshgrid(self.unique_subjects, self.unique_tasks, indexing='ij')
        s = s.flatten()
        t = t.flatten()
        subjects = np.tile(s, num_rounds)
        tasks = np.tile(t, num_rounds)
        return subjects, tasks, (num_rounds, len(self.unique_subjects), len(self.unique_tasks),)

    def sample_pairwise_contrastive(self, batch_size):
        num_rounds = 2
        square_size = int(math.sqrt(batch_size//2))
        # Sample square_size subjects and tasks
        s = self.rng.choice(self.unique_subjects, square_size)
        t = self.rng.choice(self.unique_tasks, square_size)
        s, t = np.meshgrid(s, t, indexing='ij')
        s = s.flatten()
        t = t.flatten()
        subjects = np.tile(s, num_rounds)
        tasks = np.tile(t, num_rounds)
        return subjects, tasks, (num_rounds, square_size, square_size,)


    def sample_num_trials(self, min_trials, s, t, target=False, size=None, max_num_trials=None):
        if max_num_trials is None:
            max_num_trials = self.get_num_samples(s, t, target=target)
        if self.sample_method == 'weighted':
            task_idx = self.weight_task_to_task_index[t]
            weights = self.trial_weights[task_idx, min_trials-1:max_num_trials]
            possible_num_trials = np.arange(min_trials, max_num_trials+1)
            return self.rng.choice(possible_num_trials, p=weights/weights.sum(), size=size)
        elif self.sample_method == 'uniform':
            return self.rng.integers(min_trials, max_num_trials+1, size=size)
        elif self.sample_method == 'inverse':
            possible_num_trials = np.arange(min_trials, max_num_trials+1)
            weights = 1 - 1 / possible_num_trials
            if min_trials == 1:
                weights[0] = 1/3
            return self.rng.choice(possible_num_trials, p=weights/weights.sum(), size=size)
        elif self.sample_method == 'inverted':
            possible_num_trials = np.arange(min_trials, max_num_trials+1)
            weights = 1 / possible_num_trials
            return self.rng.choice(possible_num_trials, p=weights/weights.sum(), size=size)
        else:
            raise NotImplementedError("Invalid sample method.")

    def trial_to_perc(self, num_trials, s=0, t=0, target=False, max_step=None, no_leakage=True):
        trial_max_step = max_step if max_step is not None else self.get_num_samples(s, t, target=target)
        return (self.num_samples-1)*num_trials/trial_max_step

    def sample(self, subjects, tasks):
        input_samples = []
        target_samples = []
        input_perc = []
        target_perc = []
        input_num_trials = []
        target_num_trials = []
        for s, t in zip(subjects, tasks):
            if self.input_data == 'single':
                input_num_trials_n = 1
                input_perc_n = 0
            elif self.input_data == 'bootstrap':
                input_num_trials_n = self.sample_num_trials(1, s, t, target=False)
                input_perc_n = int(self.trial_to_perc(input_num_trials_n, s, t, target=False))
            else:
                raise ValueError("Invalid input type.")
            input_indices = self.rng.choice(self.get_indices(s, t, target=False), input_num_trials_n, replace=True)
            input_bootstrap = self.data[input_indices].mean(axis=0)
            input_samples.append(input_bootstrap)
            input_perc.append(input_perc_n)
            input_num_trials.append(input_num_trials_n)

            if self.target_data == 'bootstrap_erp':
                target_num_trials_n = self.get_num_samples(s, t, target=True)
                target_perc_n = self.num_samples-1
            elif self.target_data == 'bootstrap':
                target_num_trials_n = self.sample_num_trials(input_num_trials_n, s, t, target=True)
                target_perc_n = int(self.trial_to_perc(target_num_trials_n, s, t, target=True))
            elif self.target_data == 'autoencoder':
                continue
            else:
                raise ValueError("Invalid target type.")
            target_indices = self.rng.choice(self.get_indices(s, t, target=True), target_num_trials_n, replace=True)
            target_bootstrap = self.data[target_indices].mean(axis=0)
            target_samples.append(target_bootstrap)
            target_perc.append(target_perc_n)
            target_num_trials.append(target_num_trials_n)

        input_samples = torch.stack(input_samples, dim=0)
        input_perc = torch.tensor(input_perc, dtype=torch.long)
        input_num_trials = torch.tensor(input_num_trials, dtype=torch.long)
        if self.target_data != 'autoencoder':
            target_samples = torch.stack(target_samples, dim=0)
            target_perc = torch.tensor(target_perc, dtype=torch.long)
            target_num_trials = torch.tensor(target_num_trials, dtype=torch.long)
        else:
            target_samples = input_samples
            target_perc = input_perc
            target_num_trials = input_num_trials
        return ERPSample(x=input_samples, targets=target_samples, x_perc=input_perc, target_perc=target_perc, x_trials=input_num_trials, target_trials=target_num_trials)

    def num_trials_to_cond(self, cond_type, num_trials, s, t, target=False):
        if cond_type == 'perc':
            return self.trial_to_perc(num_trials, s, t, target)
        elif cond_type == 'step':
            return num_trials - 1
        else:
            raise ValueError("Invalid condition type.")

    def interpolate(self, x, points):
        points = torch.tensor(points, device=self.data.device, dtype=torch.float)
        points_floor = points.floor().long()
        points_ceil = points.ceil().long().clamp(0, x.size(0)-1)
        points_frac = points - points_floor
        x_floor = x[points_floor]
        x_ceil = x[points_ceil]
        x_interp = x_floor + (x_ceil - x_floor) * points_frac.unsqueeze(-1).unsqueeze(-1)
        return x_interp, points+1


    def sample_trajectory(self, subjects, tasks, trajectory_size=2, trajectory_spacing='uniform', use_full=True, interpolates=False, data_target=False):
        input_samples = []
        input_perc = []
        input_num_trials = []
        target_samples = []
        target_std_samples = []

        for s, t in zip(subjects, tasks):
            all_indices = self.get_indices(s, t, use_full=use_full)
            num_samples = max(self.get_num_samples(s, t, target=False), trajectory_size, min(len(all_indices), 4))
            input_indices = self.rng.choice(all_indices, num_samples, replace=True)
            points = self.get_sample_points(s, t, trajectory_size, trajectory_spacing, end=num_samples)
            num_trials = torch.arange(1, num_samples+1, device=self.data.device)

            erp_t = self.data[input_indices].cumsum(dim=0) / num_trials.unsqueeze(-1).unsqueeze(-1)
            if data_target:
                target_indices = self.rng.choice(all_indices, 2000, replace=True)
                target_data = self.normalize(self.data[target_indices])
                erp_1_std, erp_1 = torch.std_mean(target_data, dim=0)
            if interpolates:
                erp_t, num_trials = self.interpolate(erp_t, points)
            else:
                points = points.astype(int)
                erp_t = erp_t[points]
                num_trials = num_trials[points]
            trial_perc = self.trial_to_perc(num_trials, s, t).long()

            input_samples.append(erp_t)
            input_perc.append(trial_perc)
            input_num_trials.append(num_trials)
            if data_target:
                target_samples.append(erp_1)
                target_std_samples.append(erp_1_std)


        input_samples = torch.stack(input_samples, dim=0)
        input_perc = torch.stack(input_perc, dim=0)
        input_num_trials = torch.stack(input_num_trials, dim=0)
        if data_target:
            target_samples = torch.stack(target_samples, dim=0)
            target_std_samples = torch.stack(target_std_samples, dim=0)
            print(target_std_samples.mean())
            target_dist = td.Normal(target_samples, target_std_samples)
            return ERPSample(x=input_samples, targets=target_dist, x_perc=input_perc, target_perc=None, x_trials=input_num_trials, target_trials=None)
        else:
            return ERPSample(x=input_samples, targets=None, x_perc=input_perc, target_perc=None, x_trials=input_num_trials, target_trials=None)

    def get_sample(self, s, t, num_trials=None, target=False):
        indices = self.get_indices(s, t, target)
        num_trials = len(indices) if num_trials is None else num_trials
        sample_indices = self.rng.choice(indices, num_trials, replace=True)
        return self.data[sample_indices]

    def get_sample_points(self, s, t, trajectory_size=1, trajectory_spacing='uniform', start=0, end=None, target=False):
        if end is None:
            end = self.get_num_samples(s, t, target=target)
        if trajectory_spacing == 'uniform':
            points = self.sample_num_trials(start+1, s, t, target=target, size=trajectory_size-1, max_num_trials=end)-1
        elif trajectory_spacing == 'uniform_t':
            points = self.rng.uniform(start / end, 1, trajectory_size-1) * (end-1)
        elif trajectory_spacing == 'fill':
            points = np.linspace(start, end-1, trajectory_size, endpoint=True, dtype=int)
        elif trajectory_spacing.startswith('lognorm'):
            # Get the parameters
            mean, std = trajectory_spacing[7:].replace('(', '').replace(')', '').split(',')
            # Sample the number of points
            norm_points = self.rng.normal(float(mean), float(std), trajectory_size-1)
            logit_points = 1/(1+np.exp(-norm_points))
            # Get the points
            points = logit_points * (end-1)
        else:
            raise ValueError("Invalid trajectory spacing type.")
        if trajectory_spacing != 'fill':
            points = np.concatenate((points, [end-1]))
            points = np.sort(points)
        return points