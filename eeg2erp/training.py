import wandb
import argparse
from tqdm import tqdm
from collections import defaultdict
from functools import partial
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
torch._dynamo.config.cache_size_limit = 512
torch._dynamo.config.accumulated_cache_size_limit = 1024
torch.set_float32_matmul_precision('high')

from .data import ERPDataset, ERPBootstrapTargets
from .models import ERPUNet, ERPVAE, ERPAE, CSLPAE, ERPVQVAE, ERPGCVAE, ERPEnsemble, LossAggregator, ReconstructionLoss, KLDivergenceLoss, CrossEntropyLoss, ContrastiveLoss, PairwiseContrastiveLoss, LatentPermuteLoss, SplitLatentPermuteLoss, CodebookLoss, GaussianClusteringLoss, GaussianClusteringKLLoss
from .utils import mle_distribution, plot_bootstrap_curve, plot_r_squared_boxplots, task_names
import pandas as pd
import matplotlib.pyplot as plt


def main(args):
    if not args.restrict_sampling and args.embedding_size != args.num_samples and args.cond_type == 'step':
        print(f"Warning: Embedding size is not restricted to number of samples for unrestricted sampling with step condition.")
        args.embedding_size = args.num_samples
    print("Loading data...")
    train_dataset = ERPDataset(path=args.path, split='train', processing='simple', sample_method=args.sample_method, input_data=args.input_data, target_data=args.target_data, normalize_kind=args.data_normalize, num_samples=args.num_samples, no_leakage=False, restricted=args.restrict_sampling, num_classes=args.num_classes).to('cuda')
    dev_dataset = ERPDataset(path=args.path, split='dev', processing='simple', sample_method=args.sample_method, input_data=args.input_data, target_data=args.target_data, normalize_kind=args.data_normalize, num_samples=args.num_samples, no_leakage=False, restricted=args.restrict_sampling, num_classes=args.num_classes).to('cuda')
    dev_bootstrap_targets = ERPBootstrapTargets(args.path, split='dev', processing='simple', repeats=args.repeats, input_type=args.cond_type, prominent_channel_only=args.prominent_channel_only).to('cuda')

    # Choose batch function
    if args.sampling == 'random':
        batch_fn = train_dataset.sample_batch
        dev_batch_fn = dev_dataset.sample_batch
    elif args.sampling == 'pairwise':
        batch_fn = train_dataset.sample_pairwise
        dev_batch_fn = dev_dataset.sample_pairwise
    elif args.sampling == 'contrastive':
        batch_fn = train_dataset.sample_contrastive
        dev_batch_fn = dev_dataset.sample_contrastive
        if args.batch_size < train_dataset.num_subjects * train_dataset.num_tasks * 2:
            if args.contrastive_loss == 'contrastive':
                raise ValueError(f"Contrastive loss requires contrastive sampling and at least 2 samples in a batch per subject-task pair. Try setting batch_size to {train_dataset.num_subjects * train_dataset.num_tasks * 2}.")
            print("Warning: Need at least 2 samples in a batch per subject-task pair for contrastive sampling.")
    elif args.sampling == 'pairwise_contrastive':
        batch_fn = train_dataset.sample_pairwise_contrastive
        dev_batch_fn = dev_dataset.sample_pairwise_contrastive
        if args.batch_size < 8:
            raise ValueError(f"Pairwise contrastive loss requires at least 4 samples in a batch. Try setting batch_size to 4 or above.")
    else:
        raise ValueError(f"Sampling method {args.sampling} not recognized.")

    # Choose sample function
    if args.embedding_type == 'disabled':
        encoder_cond = False
        decoder_cond = False
    elif args.input_data == 'single' and args.target_data == 'bootstrap':
        encoder_cond = False
        decoder_cond = True
    elif args.input_data == 'bootstrap' and args.target_data == 'bootstrap':
        encoder_cond = True
        decoder_cond = True
    elif args.input_data == 'single' and args.target_data == 'bootstrap_erp':
        encoder_cond = False
        decoder_cond = False
    elif args.input_data == 'bootstrap' and args.target_data == 'bootstrap_erp':
        encoder_cond = True
        decoder_cond = False
    elif args.input_data == 'bootstrap' and args.target_data == 'autoencoder':
        encoder_cond = True
        decoder_cond = True
    elif args.input_data == 'single' and args.target_data == 'autoencoder':
        encoder_cond = False
        decoder_cond = False
    else:
        raise ValueError(f"Input data {args.input_data} and target data {args.target_data} combination not supported.")

    print("Training model with:")
    print("Encoder condition:", encoder_cond)
    print("Decoder condition:", decoder_cond)
    print("Number tasks:", train_dataset.num_tasks)
    print("Number subjects:", train_dataset.num_subjects)
    print("Smallest batch size:", train_dataset.num_subjects * train_dataset.num_tasks * 2)

    def train_sample(batch_size):
        subjects, tasks, batch_shape = batch_fn(batch_size)
        erp_sample = train_dataset.sample(subjects, tasks)
        erp_sample.to('cuda')
        return erp_sample.x, erp_sample.targets, erp_sample.x_perc, erp_sample.target_perc, erp_sample.x_trials, erp_sample.target_trials, subjects, tasks, batch_shape

    def dev_sample(batch_size):
        subjects, tasks, batch_shape = dev_batch_fn(batch_size)
        erp_sample = dev_dataset.sample(subjects, tasks)
        erp_sample.to('cuda')
        return erp_sample.x, erp_sample.targets, erp_sample.x_perc, erp_sample.target_perc, erp_sample.x_trials, erp_sample.target_trials, subjects, tasks, batch_shape

    print("Initializing model...")
    if args.embedding_type == 'learned' and args.embedding_size != args.channels:
        print("Warning: Embedding size must be equal to number of channels for learned embedding.")
        args.embedding_size = args.channels

    model_factory = {
        'context_size': args.context_size,
        'in_channels': args.in_channels,
        'channels': args.channels,
        'latent_dim': args.latent_dim,
        'num_layers': args.num_layers,
        'activation': args.activation,
        'std_model': args.std_model,
        'num_samples': args.num_samples,
        'embedding_type': args.embedding_type,
        'embedding_size': args.embedding_size,
        'cosine_norm': args.cosine_norm,
        'cosine_reverse_proj': args.cosine_reverse_proj,
        'encoder_cond': encoder_cond,
        'decoder_cond': decoder_cond,
        'interpolated_residual': args.interpolated_residual
    }

    if args.variational_model == 'vae':
        arch = ERPVAE(**model_factory)
    elif args.variational_model == 'ae':
        arch = ERPAE(**model_factory)
    elif args.variational_model == 'ensemble':
        arch = ERPEnsemble(**model_factory, num_decoders=args.num_decoders)
    elif args.variational_model == 'unet':
        arch = ERPUNet(**model_factory)
    elif args.variational_model == 'vq-vae':
        arch = ERPVQVAE(**model_factory, num_clusters=args.num_clusters)
    elif args.variational_model == 'gc-vae':
        arch = ERPGCVAE(**model_factory, num_clusters=args.num_clusters)
    elif args.variational_model == 'cslp-ae':
        arch = CSLPAE(**model_factory)
    else:
        raise ValueError(f"Variational model {args.variational_model} not recognized.")

    loss_module_dict = {'recon_loss': ReconstructionLoss()}
    loss_modules = nn.ModuleList([loss_module_dict['recon_loss']])
    if args.contrastive_loss == 'contrastive':
        if args.sampling != 'contrastive' and args.sampling != 'pairwise_contrastive':
            raise ValueError("Contrastive loss requires contrastive or pairwise contrastive sampling.")
        loss_module_dict['contrastive'] = ContrastiveLoss(args.latent_dim, arch.l_ctx_size, norm=args.contrastive_norm, proj=args.contrastive_proj)
        loss_modules.append(loss_module_dict['contrastive'])
    elif args.contrastive_loss == 'cross_entropy':
        loss_module_dict['cross_entropy'] = CrossEntropyLoss(args.latent_dim, arch.l_ctx_size, train_dataset.unique_subjects, train_dataset.unique_tasks, norm=args.contrastive_norm, weight_norm=args.cross_entropy_weight_norm)
        loss_modules.append(loss_module_dict['cross_entropy'])
    elif args.contrastive_loss == 'pairwise':
        loss_module_dict['pairwise'] = PairwiseContrastiveLoss(args.latent_dim, arch.l_ctx_size, norm=args.contrastive_norm)
        loss_modules.append(loss_module_dict['pairwise'])
    if args.permutation_loss == 'latent':
        loss_module_dict['latent_permute'] = LatentPermuteLoss(args.latent_dim, arch.l_ctx_size, args.latent_target_permute)
        loss_modules.append(loss_module_dict['latent_permute'])
    elif args.permutation_loss == 'split_latent':
        loss_module_dict['split_latent_permute'] = SplitLatentPermuteLoss(args.latent_target_permute)
        loss_modules.append(loss_module_dict['split_latent_permute'])

    # Variational model specific losses
    if args.variational_model == 'vae':
        loss_module_dict['kld'] = KLDivergenceLoss()
        loss_modules.append(loss_module_dict['kld'])
    elif args.variational_model == 'vq-vae':
        loss_module_dict['codebook'] = CodebookLoss()
        loss_modules.append(loss_module_dict['codebook'])
    elif args.variational_model == 'gc-vae' and args.gaussian_clustering == 'likelihood':
        loss_module_dict['gaussian_clustering'] = GaussianClusteringLoss()
        loss_modules.append(loss_module_dict['gaussian_clustering'])
    elif args.variational_model == 'gc-vae' and args.gaussian_clustering == 'posterior':
        loss_module_dict['gaussian_clustering'] = GaussianClusteringLoss()
        loss_modules.append(loss_module_dict['gaussian_clustering'])
        loss_module_dict['gaussian_clustering_kl'] = GaussianClusteringKLLoss()
        loss_modules.append(loss_module_dict['gaussian_clustering_kl'])


    model = LossAggregator(arch, loss_modules).cuda()
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate) #, eps=1e-15
    num_params = sum(p.numel() for p in model.parameters())
    print("Model parameters:", num_params)

    if args.swa:
        @torch.no_grad()
        def ema_update(ema_param_list, current_param_list, _):
            if torch.is_floating_point(ema_param_list[0]):
                torch._foreach_lerp_(ema_param_list, current_param_list, 1 - optimizer.param_groups[0]['betas'][0])
        swa_model = optim.swa_utils.AveragedModel(model, device='cuda', multi_avg_fn=ema_update)

    if args.compile == 1:
        model.compile()
    elif args.compile > 1:
        if args.permutation_loss != 'none':
            compile_options = torch._inductor.list_mode_options('max-autotune')
            model.model.encode = torch.compile(model.model.encode, options=compile_options, fullgraph=True, dynamic=False)
            if args.variational_model != 'ensemble':
                model.model.decode = torch.compile(model.model.decode, options=compile_options, fullgraph=True, dynamic=False)
            else:
                for d in model.model.decoders:
                    d.compile(mode='max-autotune', fullgraph=True, dynamic=False)

        else:
            model.compile(mode='max-autotune', fullgraph=True, dynamic=False)

    loss_weights = defaultdict(lambda: 1.0)
    loss_weights['kld_w'] = args.kld_w
    loss_weights['recon_w'] = args.recon_w
    if args.latent_permute_w >= 0:
        loss_weights['latent_permute_w'] = args.latent_permute_w
    else:
        loss_weights['latent_permute_w'] = args.recon_w
    if args.variational_model in ['vq-vae', 'gc-vae']:
        loss_weights['commitment_cost'] = args.commitment_cost

    #Warm-up:
    print("Warm-up...")
    model.train()
    for _ in range(5):
        if args.compile > 1:
            torch.compiler.cudagraph_mark_step_begin()
        x, targets, x_perc_idx, targets_perc_idx, x_trials_idx, targets_trials_idx, subjects, tasks, batch_shape = train_sample(args.batch_size)
        if args.cond_type == 'perc':
            enc_cond = x_perc_idx
            dec_cond = targets_perc_idx
        elif args.cond_type == 'step':
            enc_cond = x_trials_idx-1
            dec_cond = targets_trials_idx-1
        effective_batch_size = x.size(0) + targets.size(0)
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            x = train_dataset.normalize(x)
            targets = train_dataset.normalize(targets)
            loss, loss_dict = LossAggregator.compute_loss(model, x, targets, subjects, tasks, enc_cond=enc_cond, dec_cond=dec_cond, batch_shape=batch_shape, loss_weights=loss_weights)
        loss.backward()
    optimizer.zero_grad(set_to_none=True)

    batches = (args.epochs * len(train_dataset)) // effective_batch_size
    ext_batches = batches + (args.ext_epochs * len(train_dataset)) // effective_batch_size
    epoch_batches = len(train_dataset) // effective_batch_size
    print("Effective batch size:", effective_batch_size)
    print("Batches:", ext_batches)
    print("Batches per epoch:", epoch_batches)
    print("Logging every:", epoch_batches * args.wandb_interval, 'or', args.wandb_interval, 'epochs')
    print("Evaluating every:", epoch_batches * args.test_interval, 'or', args.test_interval, 'epochs')

    full_test_interval_list = [int(i.strip()) for i in args.full_test_interval.split(',')]
    full_test_interval_repeat = full_test_interval_list[-1] - full_test_interval_list[-2]
    print("Full test intervals:", full_test_interval_list)
    print("Full test repeating every:", full_test_interval_repeat, 'epochs')

    if args.dry_run:
        return

    kld_ws = np.linspace(args.kld_w, args.kld_w_final, int((1-args.kld_w_pct_start) * batches))
    div_factor = args.max_lr / args.learning_rate


    linear_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1/args.final_div_factor, end_factor=1.0, total_iters=args.warmup_epochs*epoch_batches)
    cycle_scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        div_factor=div_factor,
        max_lr=args.max_lr,
        steps_per_epoch=1,
        epochs=batches-args.warmup_epochs*epoch_batches,
        three_phase=False,
        pct_start=args.pct_start,
        final_div_factor=args.final_div_factor,
        base_momentum=args.base_momentum,
        max_momentum=args.max_momentum
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [linear_scheduler, cycle_scheduler], [args.warmup_epochs*epoch_batches])

    # Evaluator functions
    if args.input_data == 'single':
        def eval_predict_fn(model, x, step, s, t):
            x.cuda()
            enc_cond = torch.full((x.size(0),), 0, device=x.device) # encoder takes single trials
            if args.target_type == 'train':
                target_step = max(train_dataset.get_num_samples(train_s, t, target=False) for train_s in train_dataset.unique_subjects)
            elif args.target_type == 'test':
                target_step = dev_dataset.get_num_samples(s, t, target=False)
            else:
                target_step = args.num_samples
            dec_cond = torch.full((x.size(0),), target_step-1, dtype=torch.long, device=x.device) # decoder targets max number of trials
            #Collapse first two dimensions
            N1, N2, C, T = x.shape
            x = x.flatten(0, 1)
            x = train_dataset.normalize(x)
            total_batchez = N1*N2
            mean = torch.zeros(total_batchez, C, T, device=x.device)
            std = torch.zeros(total_batchez, C, T, device=x.device)
            for b in range(0, total_batchez, args.batch_size):
                if args.compile > 1:
                    torch.compiler.cudagraph_mark_step_begin()
                q, z, p = model(x[b:b+args.batch_size], enc_cond[b:b+args.batch_size], dec_cond[b:b+args.batch_size])
                m = p.mean
                s = p.stddev
                mean[b:b+args.batch_size] = m
                std[b:b+args.batch_size] = s
            if b+args.batch_size < total_batchez:
                if args.compile > 1:
                    torch.compiler.cudagraph_mark_step_begin()
                q, z, p = model(x[b+args.batch_size:], enc_cond[b+args.batch_size:], dec_cond[b+args.batch_size:])
                m = p.mean
                s = p.stddev
                mean[b+args.batch_size:] = m
                std[b+args.batch_size:] = s
            mean = mean.unflatten(0, (N1, N2))
            std = std.unflatten(0, (N1, N2))
            # Perform weighted average of bootstrap samples, weighted by std**(-2)
            std_inv_sq = std**(-2)
            x_pred = (mean * std_inv_sq).sum(dim=1) / std_inv_sq.sum(dim=1)
            x_pred = train_dataset.denormalize(x_pred)
            return x_pred
    elif args.input_data == 'bootstrap':
        def eval_predict_fn(model, x, step, s, t):
            x.cuda()
            if args.compile > 1:
                torch.compiler.cudagraph_mark_step_begin()
            if args.target_type == 'train':
                target_step = max(train_dataset.get_num_samples(train_s, t, target=False) for train_s in train_dataset.unique_subjects)
            elif args.target_type == 'test':
                target_step = dev_dataset.get_num_samples(s, t, target=False)
            else:
                target_step = args.num_samples
            input_step = step
            input_perc = int(dev_dataset.trial_to_perc(step+1, max_step=target_step))
            input_cond = input_step if args.cond_type == 'step' else input_perc
            enc_cond = torch.full((x.size(0),), input_cond, device=x.device) # encoder takes bootstrap trials
            dec_cond = torch.full((x.size(0),), target_step-1, dtype=torch.long, device=x.device) # decoder targets max number of trials
            x = train_dataset.normalize(x)
            if args.variational_model == 'ensemble':
                q, z, p = model(x, enc_cond, dec_cond, full=True)
                p = mle_distribution(p)
            else:
                q, z, p = model(x, enc_cond, dec_cond)
            x_pred = train_dataset.denormalize(p.mean)
            return x_pred

    print("Initializing wandb...")
    wandb_config = dict(vars(args))
    wandb_config['batches'] = batches
    wandb_config['ext_batches'] = ext_batches
    wandb_config['div_factor'] = div_factor
    wandb_config['num_params'] = num_params
    wandb_config['encoder_cond'] = encoder_cond
    wandb_config['decoder_cond'] = decoder_cond
    wandb_config['effective_batch_size'] = effective_batch_size
    wandb_config['epoch_batches'] = epoch_batches


    group = args.group if args.group else None
    name = f"{args.variational_model}-{np.random.randint(0, 1000):03d}"

    for k, v in wandb_config.items():
        print(f"{k}: {v}")
    if group:
        print(f"Group: {group}")
    print(f"Name: {name}")

    wandb.init(project='eeg2erp', config=wandb_config, group=group, name=name)
    wandb.run.log_code(include_fn=lambda path: path.endswith("train.py") or path.endswith("model.py") or path.endswith("data.py") or path.endswith("util.py"))

    print("Training model...")
    best_test_loss = float('inf')
    best_r_squared_metric = float('-inf')
    best_test_plot_dict = None
    best_test_r_squared_dict = None
    best_loss_dict = None
    try:
        for b in (t_bar := tqdm(range(1, ext_batches+1), unit_scale=effective_batch_size, disable=not args.tqdm)):
            if args.variational_model == 'vae' and b > args.kld_w_pct_start * batches:
                kld_i = b - int(args.kld_w_pct_start * batches) - 1
                loss_weights['kld_w'] = kld_ws[kld_i] if kld_i < len(kld_ws) else args.kld_w_final
            if args.std_anneal == 'kld':
                loss_weights['std_interp'] = min(max(kld_ws[b-1] if b-1 < len(kld_ws) else args.kld_w_final, 0.0), 1.0)
            elif args.std_anneal == 'linear':
                loss_weights['std_interp'] = min(max((b-1) / batches, 0.0), 1.0)
            elif args.std_anneal == 'half':
                loss_weights['std_interp'] = min(max(2 * (b-1) / batches, 0.0), 1.0)
            elif args.std_anneal == 'abrupt': #switch to 1 after half way point
                if b < batches // 2:
                    loss_weights['std_interp'] = 0.0
                else:
                    loss_weights['std_interp'] = 1.0
            elif args.std_anneal == 'none':
                loss_weights['std_interp'] = 1.0
            elif args.std_anneal == 'disabled':
                loss_weights['std_interp'] = 0.0

            if args.compile > 1:
                torch.compiler.cudagraph_mark_step_begin()
            model.train()
            x, targets, x_perc_idx, targets_perc_idx, x_trials_idx, targets_trials_idx, subjects, tasks, batch_shape = train_sample(args.batch_size)
            if args.cond_type == 'perc':
                enc_cond = x_perc_idx
                dec_cond = targets_perc_idx
            elif args.cond_type == 'step':
                enc_cond = x_trials_idx-1
                dec_cond = targets_trials_idx-1
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                x = train_dataset.normalize(x)
                targets = train_dataset.normalize(targets)
                loss, loss_dict = LossAggregator.compute_loss(model, x, targets, subjects, tasks, enc_cond=enc_cond, dec_cond=dec_cond, batch_shape=batch_shape, loss_weights=loss_weights)
            loss.backward()
            optimizer.step()
            if b < batches:
                scheduler.step()
            print_dict = {k: v.item() for k, v in loss_dict.items()}
            print_dict['b'] = b
            print_dict['e'] = b / epoch_batches
            print_dict['s'] = b * effective_batch_size
            t_bar.set_postfix({k: v for k, v in print_dict.items() if not any(w in k for w in ['temp', 'std'])})

            if args.swa and b >= int(args.pct_start * batches):
                swa_model.update_parameters(model)

            wandb_log = {}
            with torch.inference_mode():
                if b % (epoch_batches * args.wandb_interval) == 0:
                    # Log training stats
                    wandb_log.update({
                        'b': b,
                        'e': b / epoch_batches,
                        's': b * effective_batch_size,
                        'train/b': b,
                        'train/e': b / epoch_batches,
                        'train/s': b * effective_batch_size,
                        'dev/b': b,
                        'dev/e': b / epoch_batches,
                        'dev/s': b * effective_batch_size,
                        'train/lr': optimizer.param_groups[0]['lr'],
                        'train/momentum': optimizer.param_groups[0]['betas'][0],
                    })
                    if args.variational_model != 'cslp-ae':
                        wandb_log['train/decoder_stddev'] = F.softplus(model.model.decoder_stddev).item() if args.variational_model != 'ensemble' else np.mean([F.softplus(decoder.decoder_stddev).item() for decoder in model.model.decoders])
                    if args.variational_model == 'vae':
                        wandb_log.update({
                            'train/encoder_stddev': F.softplus(model.model.encoder_stddev).item(),
                        })
                    if args.embedding_type == 'cosine':
                        if encoder_cond:
                            wandb_log['train/enc_emb_scale'] = model.model.encoder_cond.embedding_scale.exp().item()
                        if decoder_cond:
                            wandb_log['train/dec_emb_scale'] = model.model.decoder_cond.embedding_scale.exp().item() if args.variational_model != 'ensemble' else np.mean([decoder.decoder_cond.embedding_scale.exp().item() for decoder in model.model.decoders])
                    wandb_log.update({f'train/{k}': v for k, v in loss_weights.items()})
                    wandb_log.update({f'train/{k}': v for k, v in print_dict.items()})

                    # Log test loss
                    test_model = model
                    if args.swa and b >= int(args.pct_start * batches):
                        test_model = swa_model.module
                    test_model.eval()
                    dev_dataset.deterministic()
                    dev_loss_dict = defaultdict(float)
                    if b == epoch_batches or b % (epoch_batches * args.test_interval) == 0:
                        # Evaluate on random trials from dev set
                        for _ in range(args.test_batches):
                            if args.compile > 1:
                                torch.compiler.cudagraph_mark_step_begin()
                            x, targets, x_perc_idx, targets_perc_idx, x_trials_idx, targets_trials_idx, subjects, tasks, batch_shape = dev_sample(dev_dataset.num_subjects * dev_dataset.num_tasks * 2)
                            if args.cond_type == 'perc':
                                enc_cond = x_perc_idx
                                dec_cond = targets_perc_idx
                            elif args.cond_type == 'step':
                                enc_cond = x_trials_idx-1
                                dec_cond = targets_trials_idx-1
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                x = train_dataset.normalize(x)
                                targets = train_dataset.normalize(targets)
                                loss, loss_dict = LossAggregator.compute_loss(test_model, x, targets, subjects, tasks, enc_cond=enc_cond, dec_cond=dec_cond, batch_shape=batch_shape, loss_weights=loss_weights)
                            for k, v in loss_dict.items():
                                dev_loss_dict[k] += v.item() / args.test_batches
                        for k, v in dev_loss_dict.items():
                            if not any(w in k for w in ['temp', 'std']):
                                print(f"Test {k}: {v}")
                    if (b//epoch_batches in full_test_interval_list) or b % (epoch_batches * full_test_interval_repeat) == 0:
                        # Evaluate on dev bootstrap targets
                        eval_predict_fn_ = partial(eval_predict_fn, test_model)
                        plot_dict, r_squared_dict, predictions, bootstrap_total_loss = dev_bootstrap_targets.get_loss(eval_predict_fn_, F.mse_loss, dataset=dev_dataset, single_in=args.input_data=='single', tqdm_disabled=not args.tqdm)
                        df_rsquared = pd.DataFrame([(s, task_names[t], v, name) for s in r_squared_dict.keys() for t, v in r_squared_dict[s].items()], columns=['Subject', 'Task', 'R^2', 'Method'])
                        r_squared_metric = df_rsquared['R^2'].mean()

                        final_losses = []
                        for s in plot_dict.keys():
                            for t in plot_dict[s].keys():
                                final_losses.append(plot_dict[s][t][-1])
                        final_bootstrap_loss = np.mean(final_losses) * 1e12

                        dev_loss_dict['r_squared_mean'] = r_squared_metric
                        dev_loss_dict['bootstrap_loss'] = bootstrap_total_loss
                        dev_loss_dict['final_bootstrap_loss'] = final_bootstrap_loss

                        # Save model if best
                        should_save = False
                        if args.save_objective == 'r_squared' or args.target_data == 'autoencoder':
                            should_save = r_squared_metric > best_r_squared_metric
                        elif args.save_objective == 'loss':
                            should_save = dev_loss_dict['loss'] < best_test_loss
                        if should_save:
                            best_test_loss = dev_loss_dict['loss']
                            best_r_squared_metric = r_squared_metric
                            best_test_plot_dict = plot_dict
                            best_test_r_squared_dict = r_squared_dict
                            best_loss_dict = dev_loss_dict
                            fig1, axs1 = plot_bootstrap_curve(b / epoch_batches, args.path, dev_dataset.unique_subjects, dev_dataset.unique_tasks, args.repeats, dev_bootstrap_targets.steps, (name, best_test_plot_dict))
                            fig2, axs2 = plot_r_squared_boxplots(b / epoch_batches, args.path, args.repeats, df_rsquared)
                            wandb.log({"dev/plot": wandb.Image(fig1), "dev/r_squared": wandb.Image(fig2)})
                            plt.close('all')
                            print("Saving model...")
                            torch.save(model.state_dict(), f"{args.path}Models/{name}.pt")
                            torch.save(wandb_config, f"{args.path}Models/{name}.config")
                            if args.swa and b >= int(args.pct_start * batches):
                                torch.save(swa_model.state_dict(), f"{args.path}Models/{name}_swa.pt")
                            torch.save({'plot': best_test_plot_dict, 'r_squared': best_test_r_squared_dict}, f"{args.path}Models/{name}_test_results.pt")

                        print(f"Test r_squared: {r_squared_metric:.4g}; Best r_squared: {best_r_squared_metric:.4g}, Bootstrap Loss: {bootstrap_total_loss:.4g}, Best Loss: {best_test_loss:.4g}")
                    wandb_log.update({f'dev/{k}': v for k, v in dev_loss_dict.items()})
                    wandb.log(wandb_log)
    finally:
        if best_loss_dict is not None:
            # Log best test loss
            final_log = {'dev/best_r_squared': best_r_squared_metric, "dev/best_loss": best_test_loss}
            # Print only some of the best loss dict
            for k in ['recon_loss', 'loss']:
                if k in best_loss_dict:
                    final_log[f"dev/best_{k}"] = best_loss_dict[k]
            wandb.log(final_log)
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--path', type=str, default='data/')
    parser.add_argument('--compile', type=int, default=2)
    parser.add_argument("--tqdm", type=int, default=1)

    # Training and hyperparameters
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--ext_epochs", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1568) #784 min for contrastive, 1568 seems to be sweet spot
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--channels", type=int, default=256) #try 512
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--kld_w", type=float, default=0.0) #0.0
    parser.add_argument('--kld_w_pct_start', type=float, default=0.0) #0.3
    parser.add_argument('--kld_w_final', type=float, default=1.0) #1.0
    parser.add_argument('--std_anneal', type=str, default='half') #(half) - none, kld, linear, half, abrupt
    parser.add_argument("--std_model", type=str, default='full') #(full) - channel, full
    parser.add_argument("--recon_w", type=float, default=1.0)
    parser.add_argument("--embedding_type", type=str, default='cosine') #(cosine) - cosine, learned, disabled
    parser.add_argument("--disable_embedding", type=int, default=0) #0
    parser.add_argument('--cosine_norm', type=str, default='standard') #(standard) - standard, l2, none
    parser.add_argument('--cosine_reverse_proj', type=int, default=1) #1
    parser.add_argument('--embedding_size', type=int, default=256) #585 for steps, 256 for perc
    parser.add_argument('--interpolated_residual', type=str, default='residual_interp') #(residual_interp?) - none, global_interp, local_interp, residual_interp
    parser.add_argument('--num_decoders', type=int, default=5) #5
    parser.add_argument('--num_classes', type=int, default=4) #4

    # Data
    parser.add_argument('--context_size', type=int, default=256)
    parser.add_argument('--in_channels', type=int, default=30)
    parser.add_argument('--input_data', type=str, default='bootstrap') #(bootstrap) - single, bootstrap
    parser.add_argument('--target_data', type=str, default='bootstrap') # (bootstrap?) - bootstrap, bootstrap_erp
    parser.add_argument('--data_normalize', type=str, default='standard') #standard, scale, none
    parser.add_argument('--cond_type', type=str, default='step') #(perc?) - step, perc
    parser.add_argument('--target_type', type=str, default='test') #max, train, test
    parser.add_argument('--no_leakage', type=int, default=1) #0
    parser.add_argument('--sampling', type=str, default='contrastive') #random, pairwise, contrastive, pairwise_contrastive
    parser.add_argument('--num_samples', type=int, default=585) #585 for all
    parser.add_argument('--restrict_sampling', type=int, default=1) #0
    parser.add_argument('--sample_method', type=str, default='uniform') #uniform, weighted

    # Model
    parser.add_argument('--variational_model', type=str, default='ae') #vae, ae, unet, vq-vae, gc-vae
    parser.add_argument('--activation', type=str, default='glu') #relu
    parser.add_argument('--contrastive_loss', type=str, default='contrastive') #none, contrastive, cross_entropy, pairwise
    parser.add_argument('--contrastive_norm', type=int, default=1)
    parser.add_argument('--contrastive_proj', type=int, default=0) #0
    parser.add_argument('--cross_entropy_weight_norm', type=int, default=1)
    parser.add_argument('--permutation_loss', type=str, default='split_latent') #(split_latent) - none, latent, split_latent
    parser.add_argument('--latent_target_permute', type=int, default=0) #1?
    parser.add_argument('--latent_permute_w', type=float, default=-1)
    parser.add_argument('--gaussian_clustering', type=str, default='likelihood') #likelihood, posterior
    parser.add_argument('--num_clusters', type=int, default=512)
    parser.add_argument('--commitment_cost', type=float, default=0.25)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=4e-4) #4e-4 stable
    parser.add_argument("--max_lr", type=float, default=4e-4)
    parser.add_argument("--pct_start", type=float, default=0.02) #0.02
    parser.add_argument("--final_div_factor", type=float, default=1e2) #1e3?
    parser.add_argument("--base_momentum", type=float, default=0.999)
    parser.add_argument("--max_momentum", type=float, default=0.9999) #0.999?
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--swa", type=int, default=0)

    # Reporting
    parser.add_argument("--test_interval", type=int, default=10)
    parser.add_argument("--full_test_interval", type=str, default='1,10,20,50,100,150,200') #'1,10,20,50,100,150,200'
    parser.add_argument("--test_batches", type=int, default=42)
    parser.add_argument("--wandb_interval", type=int, default=1)
    parser.add_argument("--group", type=str, default="")

    # Saving and evaluation
    parser.add_argument('--repeats', type=int, default=200) #200
    parser.add_argument('--prominent_channel_only', type=int, default=1) #1
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--save_objective', type=str, default='r_squared') #r_squared, loss
    args = parser.parse_args()
    main(args)