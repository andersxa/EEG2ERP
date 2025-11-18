import os
from functools import partial
from collections import defaultdict
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as td
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch.set_float32_matmul_precision('high')

from .data import ERPDataset, ERPBootstrapTargets, ERPCoreNormalizer, ModelState, R2Metric, R2WeightedMetric, MSEMetric, GaussianNLLMetric, NFEMetric, TrueVariance, PredictionVariance, ModelVariance, ModelPredictions
from .utils import load_model, mle_distribution, get_measures

import pandas as pd


def test_model(args, test_dataset=None, test_bootstrap_targets=None, normalizer=None):
    is_model = not args.model.startswith('base')
    device = 'cuda'
    group = 'none'
    if is_model:
        model_name = args.model
        if args.prepend_id and args.id:
            model_name = args.id + '_' + model_name
        if args.print_log:
            print("Loading model...")
        model_config, model = load_model(args.path, model_name)
        group = model_config['group']

        model = model.to(device)
    else:
        model_config = {
            'sample_method': 'uniform',
            'input_data': 'single',
            'target_data': 'bootstrap_erp',
            'data_normalize': 'standard',
            'num_samples': 585,
            'cond_type': 'step',
            'group': 'X'+args.model.replace('base-', '').replace('base_', '')[0].upper(),
        }
        group = model_config['group']
        if 'molina' in args.model.lower():
            device = 'cpu'
    if args.print_log:
        print("Loading data...")
    if test_dataset is None:
        test_dataset = ERPDataset(path=args.path, split=args.split, processing='simple', sample_method=model_config['sample_method'], input_data=model_config['input_data'], target_data=model_config['target_data'], normalize_kind=model_config['data_normalize'], num_samples=model_config['num_samples'], no_leakage=True, restricted=True).deterministic().to(device, data=True)
    else:
        test_dataset.reconfigure(sample_method=model_config['sample_method'], input_data=model_config['input_data'], target_data=model_config['target_data'], normalize_kind=model_config['data_normalize'], num_samples=model_config['num_samples'])
    if test_bootstrap_targets is None:
        test_bootstrap_targets = ERPBootstrapTargets(args.path, split=args.split, processing='simple', repeats=args.repeats, input_type=model_config['cond_type'], prominent_channel_only=True).to(device)
    else:
        test_bootstrap_targets.reconfigure(input_type=model_config['cond_type'])
    if normalizer is None:
        normalizer = ERPCoreNormalizer(path=args.path, processing='simple', normalize_kind=model_config['data_normalize'])
    else:
        normalizer.reconfigure(normalize_kind=model_config['data_normalize'])

    if is_model:
        num_params = sum(p.numel() for p in model.parameters())
        if args.print_log:
            print("Model parameters:", num_params)

        if args.compile == 1:
            model.compile()
        elif args.compile > 1:
            if model_config['permutation_loss'] != 'none':
                compile_options = torch._inductor.list_mode_options('max-autotune')
                model.encode = torch.compile(model.encode, options=compile_options, fullgraph=True, dynamic=False)
                model.decode = torch.compile(model.decode, options=compile_options, fullgraph=True, dynamic=False)
            else:
                model.compile(mode='max-autotune', fullgraph=True, dynamic=False)

        target_type = model_config['target_type'] if not args.override_target_type else args.override_target_type
        if args.override_target_type:
            train_num_samples_name = 'task_max_samples.pt' if not model_config['no_leakage'] else 'task_max_samples_no_leakage.pt'
            train_num_samples = torch.load(args.path+train_num_samples_name, weights_only=False)
        if model_config['input_data'] == 'single':
            def eval_predict_fn(x, step, s, t, channel):
                x.cuda()
                enc_cond = torch.full((x.size(0),), 0, device=x.device) # encoder takes single trials
                if model_config['target_type'] == 'test':
                    target_step = test_dataset.get_num_samples(s, t, target=False)
                else:
                    target_step = model_config['num_samples']
                dec_cond = torch.full((x.size(0),), target_step-1, dtype=torch.long, device=x.device) # decoder targets max number of trials
                #Collapse first two dimensions
                N1, N2, C, T = x.shape
                x = x.flatten(0, 1)
                x = normalizer.normalize(x)
                total_batchez = N1*N2
                mean = torch.zeros(total_batchez, C, T, device=x.device)
                std = torch.zeros(total_batchez, C, T, device=x.device)
                for b in range(0, total_batchez, model_config['batch_size']):
                    if args.compile > 1:
                        torch.compiler.cudagraph_mark_step_begin()
                    q, z, p = model(x[b:b+model_config['batch_size']], enc_cond[b:b+model_config['batch_size']], dec_cond[b:b+model_config['batch_size']])
                    m = p.mean
                    s = p.stddev
                    mean[b:b+model_config['batch_size']] = m
                    std[b:b+model_config['batch_size']] = s
                if b+model_config['batch_size'] < total_batchez:
                    if args.compile > 1:
                        torch.compiler.cudagraph_mark_step_begin()
                    q, z, p = model(x[b+model_config['batch_size']:], enc_cond[b+model_config['batch_size']:], dec_cond[b+model_config['batch_size']:])
                    m = p.mean
                    s = p.stddev
                    mean[b+model_config['batch_size']:] = m
                    std[b+model_config['batch_size']:] = s
                mean = mean.unflatten(0, (N1, N2))
                std = std.unflatten(0, (N1, N2))
                # Perform weighted average of bootstrap samples, weighted by std**(-2)
                std_inv_sq = std**(-2)
                x_pred = (mean * std_inv_sq).sum(dim=1) / std_inv_sq.sum(dim=1)
                std_pred = std_inv_sq.sum(dim=1)**(-0.5)
                x_pred = normalizer.denormalize(x_pred)
                p_pred = normalizer.denormalize_dist(td.Normal(x_pred, std_pred))
                pred = ModelState(x=x, x_hat=x_pred, p=p_pred)
                return pred
        elif model_config['input_data'] == 'bootstrap':
            def eval_predict_fn(x, step, s, t, channel):
                x.cuda()
                if args.compile > 1:
                    torch.compiler.cudagraph_mark_step_begin()
                x = x.mean(dim=1)
                if target_type == 'train':
                    target_step = train_num_samples[t]
                elif target_type == 'test':
                    target_step = test_dataset.get_num_samples(s, t, target=False)
                else:
                    target_step = model_config['num_samples']
                input_step = step
                input_perc = int(test_dataset.trial_to_perc(step+1, max_step=target_step))
                enc_cond = torch.full((x.size(0),), input_step if model_config['cond_type'] == 'step' else input_perc, device=x.device) # encoder takes bootstrap trials
                dec_cond = torch.full((x.size(0),), target_step-1, dtype=torch.long, device=x.device) # decoder targets max number of trials
                x = normalizer.normalize(x)
                if model_config['variational_model'] == 'ensemble':
                    q, z, p = model(x, enc_cond, dec_cond, full=True)
                    p = mle_distribution(p)
                else:
                    q, z, p = model(x, enc_cond, dec_cond)
                x_pred = normalizer.denormalize(p.mean)
                p_pred = normalizer.denormalize_dist(p)
                pred = ModelState(x=x, x_hat=x_pred, p=p_pred)
                return pred
    else: # Base model
        if 'leonowicz' in args.model.lower():
            def eval_predict_fn(x: torch.Tensor, steps: int, s: int, t: int, channel: int):
                x = x[..., channel, :]
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
                pred = ModelState(x=x, x_hat=x_pred)
                return pred
        elif 'kotowski' in args.model.lower():
            def eval_predict_fn(x: torch.Tensor, steps: int, s: int, t: int, channel: int):
                x = x[..., channel, :]
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
                pred = ModelState(x=x, x_hat=x_pred)
                return pred
        elif 'molina' in args.model.lower():
            from dtaidistance import dtw
            if 'fast' in args.model.lower():
                from joblib import Parallel, delayed
                def process_repeat(s_r, s_i, K):
                    s_warped = []
                    for i in range(K):
                        _, paths = dtw.warping_paths_fast(s_r, s_i[i], use_c=True)
                        path = dtw.best_path(paths)
                        s_warp, path = dtw.warp(s_r, s_i[i], path, use_c=True)
                        s_warped.append(s_warp)
                    return s_warped
                def eval_predict_fn(x: torch.Tensor, steps: int, s: int, t: int, channel: int):
                    x = x[..., channel, :]
                    # In shape: (repeats, trials, channels, timepoints)
                    r = x.mean(dim=1).cpu().numpy().astype(float)
                    x_n = x.cpu().numpy().astype(float)
                    
                    reps = Parallel(n_jobs=-1)(delayed(process_repeat)(r[repeat].copy(), x_n[repeat].copy(), x.size(1)) for repeat in range(x.size(0)))
                    
                    x_pred = torch.stack([torch.tensor(v, device=x.device, dtype=x.dtype).mean(dim=0) for v in reps])
                    pred = ModelState(x=x, x_hat=x_pred)
                    return pred
            else:
                class AbsDist(dtw.innerdistance.CustomInnerDist):
                    @staticmethod
                    def inner_dist(x, y):
                        return np.abs(x - y)
                    @staticmethod
                    def result(x):
                        return x
                    @staticmethod
                    def inner_val(x):
                        return x
                def eval_predict_fn(x: torch.Tensor, steps: int, s: int, t: int, channel: int):
                    x = x[..., channel, :]
                    # In shape: (repeats, trials, channels, timepoints)
                    r = x.mean(dim=1).cpu().numpy().astype(float)
                    x_n = x.cpu().numpy().astype(float)
                    v = []
                    for repeat in range(x.size(0)):
                        s_r = r[repeat]
                        s_i = x_n[repeat]
                        s_warped = []
                        for i in range(x.size(1)):
                            _, paths = dtw.warping_paths(s_r, s_i[i], inner_dist=AbsDist)
                            path = dtw.best_path(paths)
                            s_warp, path = dtw.warp(s_r, s_i[i], path, inner_dist=AbsDist)
                            s_warped.append(s_warp)
                        s_warped = torch.tensor(s_warped, device=x.device, dtype=x.dtype).mean(dim=0)
                        v.append(s_warped)
                    x_pred = torch.stack(v)
                    pred = ModelState(x=x, x_hat=x_pred)
                    return pred
        elif 'simple' in args.model.lower():
            def eval_predict_fn(x, step, s, t, channel):
                x = x[..., channel, :]
                x_pred = x.mean(dim=1)
                pred = ModelState(x=x, x_hat=x_pred)
                return pred
        else:
            raise ValueError(f"Model {args.model} not recognized.")
    
    metrics = [MSEMetric(single_channel=is_model), R2Metric(single_channel=is_model)]
    if is_model:
        metrics.append(GaussianNLLMetric(single_channel=is_model))
        metrics.append(R2WeightedMetric(single_channel=is_model))
    if args.split == 'test' and not args.eval_all:
        metrics.append(ModelPredictions(single_channel=is_model))
        metrics.append(TrueVariance(single_channel=is_model))
        if args.difference_waveform_measures:
            metrics.append(PredictionVariance(single_channel=is_model))
        if is_model:
            metrics.append(ModelVariance(single_channel=is_model))
    if args.print_log:
        print("Evaluating model...")
    # Log test loss
    if is_model:
        model.eval()

    df, results = evaluate_model(args, eval_predict_fn, metrics, test_dataset, test_bootstrap_targets, exclude_from_df={'Prediction'})
    
    # Aggregate means over both columns "Subject" and "Task"
    if args.print_log:
        print_dfs = [df] if not args.difference_waveform_measures else df
        groupbys = [['Subject', 'Task']] if not args.difference_waveform_measures else [['Subject', 'Task'], ['Paradigm']]
        for i, (d, groupby) in enumerate(zip(print_dfs, groupbys)):
            mean_over_all = d.groupby(groupby).mean(numeric_only=True)
            if i == 0:
                mean_over_all = mean_over_all.mean()
            print("Mean over all subjects and tasks:")
            print(mean_over_all)
    if args.save:
        if 'Prediction' in results:
            del results['Prediction']
        # Save results
        if not os.path.exists(f'{args.path}/Results/{group}/{args.split}'):
            os.makedirs(f'{args.path}/Results/{group}/{args.split}')
        print("Saving results...", f'{args.path}/Results/{group}/{args.split}/{args.model}_results*.pt')
        if not args.eval_all and not args.override_steps:
            torch.save({'df': df, 'results': results}, f'{args.path}/Results/{group}/{args.split}/{args.model}_results.pt')
        elif args.eval_all and not args.override_steps:
            torch.save({'df': df, 'results': results}, f'{args.path}/Results/{group}/{args.split}/{args.model}_results_all.pt')
        elif args.override_steps:
            o_steps = args.override_steps.replace(',', '_')
            torch.save({'df': df, 'results': results}, f'{args.path}/Results/{group}/{args.split}/{args.model}_results_{o_steps}.pt')

    return df, results

def evaluate_model(args, eval_predict_fn, metrics, test_dataset, test_bootstrap_targets, exclude_from_df=None):
    if exclude_from_df is None:
        exclude_from_df = set()
    with torch.inference_mode():
        if args.eval_all:
            steps = list(range(1, args.max_step_eval+1))
            steps_str = steps
        else:
            steps = [1, 5, 0.1, float('inf')]
            steps_str = ['K=1', 'K=5', '10%', '100%']
        if args.override_steps:
            steps = [int(s.strip()) for s in args.override_steps.split(',') if s.strip()]
            steps_str = steps
        test_dataset.deterministic()
        
        results = test_bootstrap_targets.get_results(eval_predict_fn, metrics, test_dataset, steps, tqdm_disabled=not args.print_log)
        
        subjects = test_dataset.unique_subjects
        tasks = test_dataset.unique_tasks

        
        df = get_dataframe(results, subjects, tasks, test_dataset, args.model, steps_str, exclude_from_df)

        if args.split == 'test' and args.difference_waveform_measures:
            diff_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
            for s in subjects:
                for paradigm, (t1, t2) in test_dataset.paradigm_components.items():
                    for v1, v2 in zip(results['Prediction'][s][t1], results['Prediction'][s][t2]):
                        diff_wave = v1 - v2
                        for k, v in get_measures(diff_wave, paradigm, fs=256).items():
                            diff_results[k][s][paradigm].append(v.mean().item())
            diff_data = []
            diff_metric_names = [n for n in diff_results.keys()]
            for s in subjects:
                for paradigm in test_dataset.paradigm_components.keys():
                    row = [args.model, s, paradigm]
                    for metric_name in diff_metric_names:
                        for v in diff_results[metric_name][s][paradigm]:
                            row.append(v)
                    diff_data.append(row)
            diff_columns = ['Model', 'Subject', 'Paradigm']
            for metric_name in diff_metric_names:
                for step in steps_str:
                    diff_columns.append(f"{metric_name} {step}")
            diff_df = pd.DataFrame(diff_data, columns=diff_columns)
            return (df, diff_df), results
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




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--path', type=str, default='data/')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--compile', type=int, default=2)
    parser.add_argument("--print_log", type=int, default=1)
    parser.add_argument("--id", type=str, default="")
    parser.add_argument("--prepend_id", type=int, default=0)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--difference_waveform_measures', type=int, default=0)

    # Model
    parser.add_argument("--model", type=str, default="resnet-075")
    parser.add_argument("--override_target_type", type=str, default="")

    # Saving and evaluation
    parser.add_argument('--num_points', type=int, default=25)
    parser.add_argument('--trajectory_spacing', type=str, default='fill')
    parser.add_argument('--repeats', type=int, default=200)

    parser.add_argument('--eval_all', type=int, default=0)
    parser.add_argument('--max_step_eval', type=int, default=100)
    parser.add_argument('--override_steps', type=str, default='')

    args = parser.parse_args()
    test_model(args)