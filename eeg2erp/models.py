import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td
import math
import numpy as np
from scipy import integrate
from collections import defaultdict

class GlobalPSiLU(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.beta = nn.Parameter(torch.full((channels, time_dim), 1.664007306098938))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)

class GlobalActNorm(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.act = GlobalPSiLU(channels, time_dim)
        self.norm = nn.LayerNorm([channels, time_dim])

    def forward(self, x):
        x = self.act(x)
        x = self.norm(x)
        return x

class ActNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.act = nn.ReLU()
        self.norm = nn.InstanceNorm1d(channels, affine=True)

    def forward(self, x):
        x = self.act(x)
        x = self.norm(x)
        return x

class ConvAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=1, padding=0, bias=False):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, kernel_size, bias=bias, padding=padding)
        self.act = ActNorm(dim_out)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.residual = dim_in == dim_out

    def forward(self, x):
        res = x
        x = self.proj(x)
        x = self.act(x)
        if self.residual:
            return res + x
        else:
            return x

class HalfGLU(nn.Module):
    def __init__(self, dim_in, dim_out, bias=False):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, 3, padding=1, bias=bias)
        self.proj_out = nn.Conv1d(dim_out // 2, dim_out, 1, bias=bias)
        self.norm = nn.InstanceNorm1d(dim_out // 2, affine=True)

    def forward(self, x):
        res = x
        x1, x2 = self.proj(x).chunk(2, 1)
        x1 = self.norm(x1)
        act = torch.sigmoid(x1) * x2
        out = self.proj_out(act)
        return res + out

class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, affine=True):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, 2*dim_out, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm1d(dim_out, affine=affine)
        self.residual = dim_in == dim_out

    def forward(self, x):
        res = x
        x1, x2 = self.proj(x).chunk(2, -2)
        x1 = self.norm(x1)
        if self.residual:
            return res + torch.sigmoid(x1) * x2
        return torch.sigmoid(x1) * x2

class ConvReLU(nn.Module):
    def __init__(self, dim_in, dim_out, affine=True):
        super().__init__()
        self.proj = nn.Conv1d(dim_in, dim_out, 3, padding=1, bias=False)
        self.norm = nn.InstanceNorm1d(dim_out, affine=affine)
        self.residual = dim_in == dim_out

    def forward(self, x):
        x_ = self.proj(x)
        x_ = F.relu(x_)
        x_ = self.norm(x_)
        if self.residual:
            return x + x_
        else:
            return x_

class LinearGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, 2*dim_out, bias=False)
        self.norm = nn.LayerNorm(dim_out)
        self.residual = dim_in == dim_out

    def forward(self, x):
        res = x
        x1, x2 = self.proj(x).chunk(2, -1)
        x1 = self.norm(x1)
        out = torch.sigmoid(x1) * x2
        if self.residual:
            return res + out
        return out

class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out, activation='glu', affine=True):
        super().__init__()
        self.in_channels = dim_in
        self.channels = dim_mid
        self.out_channels = dim_out

        if activation == 'glu':
            self.conv1 = GLU(dim_in, dim_mid, affine=affine)
            self.conv2 = GLU(dim_mid, dim_mid, affine=affine)
            self.conv3 = GLU(dim_mid, dim_out, affine=affine)
        elif activation == 'relu':
            self.conv1 = ConvReLU(dim_in, dim_mid, affine=affine)
            self.conv2 = ConvReLU(dim_mid, dim_mid, affine=affine)
            self.conv3 = ConvReLU(dim_mid, dim_out, affine=affine)
        else:
            raise ValueError(f"Activation {activation} not recognized.")

    def __repr__(self):
        return f'ConvBlock({self.in_channels}, {self.channels}, {self.out_channels})'

    def forward(self, x):
        # Residual before
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class GlobalInterpolation(nn.Module):
    def forward(self, x, init_res, res):
        res = F.interpolate(init_res, size=x.size(-1), mode='linear')
        x = x + res
        return x, res

class LocalInterpolation(nn.Module):
    def forward(self, x, init_res, res):
        res = F.interpolate(res, size=x.size(-1), mode='linear')
        x = x + res
        return x, res

class ResidualInterpolation(nn.Module):
    def forward(self, x, init_res, res):
        res = F.interpolate(res, size=x.size(-1), mode='linear')
        x = x + res
        res = x
        return x, res

class ResidualSequential(nn.Module):
    def __init__(self, *modules, interpolated_residual='none'):
        super(ResidualSequential, self).__init__()
        self.layers = nn.ModuleList(modules)
        self.interpolated_residual = interpolated_residual
        if self.interpolated_residual == 'global_interp':
            self.residual_interpolation = GlobalInterpolation()
        elif self.interpolated_residual == 'local_interp':
            self.residual_interpolation = LocalInterpolation()
        elif self.interpolated_residual == 'residual_interp':
            self.residual_interpolation = ResidualInterpolation()
        else:
            raise ValueError(f"Interpolated residual {self.interpolated_residual} not recognized.")

    def forward(self, x):
        res = init_res = x
        for module in self.layers:
            x = module(x)
            x, res = self.residual_interpolation(x, init_res, res)
        return x

class StridedConvolutionalEncoder(nn.Module):
    def __init__(self, channels, kernel_size, num_layers, activation='glu', interpolated_residual='none'):
        super(StridedConvolutionalEncoder, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        def get_conv_layer():
            return nn.Sequential(
                ConvBlock(channels, channels, channels, activation=activation),
                nn.Conv1d(channels, channels, kernel_size, stride=2, padding=0, bias=False),
                ActNorm(channels)
            )
        layers = [get_conv_layer() for _ in range(num_layers)]
        if interpolated_residual != 'none':
            self.conv_layers = ResidualSequential(*layers, interpolated_residual=interpolated_residual)
        else:
            self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class TransposedConvolutionalDecoder(nn.Module):

    def __init__(self, channels, kernel_size, num_layers, activation='glu', interpolated_residual='none'):
        super(TransposedConvolutionalDecoder, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        def get_conv_layer():
            return nn.Sequential(
                nn.ConvTranspose1d(channels, channels, kernel_size, stride=2, padding=0, bias=False),
                ActNorm(channels),
                ConvBlock(channels, channels, channels, activation=activation),
            )
        layers = [get_conv_layer() for _ in range(num_layers)]
        if interpolated_residual != 'none':
            self.conv_layers = ResidualSequential(*layers, interpolated_residual=interpolated_residual)
        else:
            self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        return x

def _trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    l = norm_cdf((a-mean) / std)
    u = norm_cdf((b-mean) / std)

    tensor.uniform_(2*l - 1, 2*u - 1)

    tensor.erfinv_()

    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)

    tensor.clamp_(min=a, max=b)
    return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)


class PositionalEncoding(nn.Module):

    def __init__(self, channels, num_seq):
        super(PositionalEncoding, self).__init__()
        self.channels = channels
        self.num_seq = num_seq
        self.pos_encoding = nn.Parameter(torch.empty(1, num_seq, channels))
        trunc_normal_(self.pos_encoding, std=0.02)

    def forward(self, x):
        return x + self.pos_encoding

class TransformerBottleneck(nn.Module):

    def __init__(self, channels, num_layers, num_seq, num_heads=8):
        super(TransformerBottleneck, self).__init__()
        self.channels = channels
        self.num_layers = num_layers

        def get_transformer_layer():
            return (
                PositionalEncoding(channels, num_seq),
                nn.TransformerEncoderLayer(channels, num_heads, 4*channels, 0.1, activation="gelu", batch_first=True),
            )

        self.transformer_layers = nn.Sequential(*[l for _ in range(num_layers) for l in get_transformer_layer()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.transformer_layers(x)
        x = x.permute(0, 2, 1)
        return x

class LossAggregator(nn.Module):
    def __init__(self, model, loss_modules):
        super(LossAggregator, self).__init__()
        self.model = model
        self.loss_modules = nn.ModuleList(loss_modules)

    def forward(self, *args, **kwargs):
        q, z, p = self.model(*args, **kwargs)
        return q, z, p

    @staticmethod
    def compute_loss(model, x, x_targets, subjects, tasks, enc_cond, dec_cond, batch_shape=None, loss_weights=defaultdict(lambda: 1.0)):
        q, z, p = model(x, enc_cond, dec_cond)
        if 'std_interp' in loss_weights:
            p.scale = 1 + (p.scale - 1) * loss_weights['std_interp']
        loss_dict = defaultdict(float)
        for loss_module in model.loss_modules:
            ld = loss_module(model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights)
            for k, v in ld.items():
                loss_dict[k] += v
        loss_dict['p_std'] = p.stddev.mean()
        loss = loss_dict['loss']
        return loss, loss_dict

class ReconstructionLoss(nn.Module):
    def forward(self, model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights):
        recon_loss = -p.log_prob(x_targets).mean()
        loss_dict = {
            'loss': loss_weights['recon_w'] * recon_loss,
            'recon_loss': recon_loss,
        }
        return loss_dict

class KLDivergenceLoss(nn.Module):
    def forward(self, model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights):
        prior = td.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z))
        kl_loss = td.kl_divergence(q, prior).mean()
        loss_dict = {
            'loss': loss_weights['kld_w'] * kl_loss,
            'q_std': q.stddev.mean(),
            'kl_loss': kl_loss,
        }
        return loss_dict

class LatentPermuteLoss(nn.Module):
    def __init__(self, latent_dim, l_ctx_size, target_permute=False):
        super(LatentPermuteLoss, self).__init__()
        self.channels = latent_dim*l_ctx_size
        self.latent_dim = latent_dim
        self.l_ctx_size = l_ctx_size
        self.subject_proj = LinearGLU(self.channels, 2*self.channels)
        self.task_proj = LinearGLU(self.channels, 2*self.channels)
        self.out_proj = nn.Sequential(LinearGLU(4*self.channels, self.channels), nn.Linear(self.channels, self.channels, bias=False))
        self.target_permute = target_permute

    def forward(self, model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights):
        n, s, t = batch_shape
        z = z.reshape(n, s, t, -1)
        l_s = self.subject_proj(z)
        l_t = self.task_proj(z)

        # Create permutations
        subj_perm = torch.randperm(s, device=z.device) #used to permute the subject axis
        task_perm = torch.randperm(t, device=z.device) #used to permute the task axis
        l_s = l_s[:, :, task_perm]
        l_t = l_t[:, subj_perm, :]

        if self.target_permute:
            x_targets = x_targets.unflatten(0, (n, -1))
            dec_cond = dec_cond.unflatten(0, (n, -1))
            targets_perm = torch.randperm(n, device=z.device)
            x_targets = x_targets[targets_perm].flatten(0, 1)
            dec_cond = dec_cond[targets_perm].flatten(0, 1)

        z = self.out_proj(torch.cat((l_s, l_t), dim=-1)).reshape(n*s*t, self.latent_dim, self.l_ctx_size)
        p = model.model.decode(z, dec_cond)

        latent_permute_loss = -p.log_prob(x_targets).mean()
        loss_dict = {
            'loss': loss_weights['latent_permute_w'] * latent_permute_loss,
            'latent_permute_loss': latent_permute_loss,
        }

        return loss_dict

class SplitLatentPermuteLoss(nn.Module):
    def __init__(self, target_permute=False):
        super(SplitLatentPermuteLoss, self).__init__()
        self.target_permute = target_permute

    def forward(self, model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights):
        n, s, t = batch_shape
        z_ = z.reshape(n, s, t, -1)
        l_s, l_t = z_.chunk(2, -1)

        # Create permutations
        subj_perm = torch.randperm(s, device=z.device) #used to permute the subject axis
        task_perm = torch.randperm(t, device=z.device) #used to permute the task axis
        l_s = l_s[:, :, task_perm]
        l_t = l_t[:, subj_perm, :]

        if self.target_permute:
            x_targets = x_targets.unflatten(0, (n, -1))
            dec_cond = dec_cond.unflatten(0, (n, -1))
            targets_perm = torch.randperm(n, device=z.device)
            x_targets = x_targets[targets_perm].flatten(0, 1)
            dec_cond = dec_cond[targets_perm].flatten(0, 1)

        z = torch.cat((l_s, l_t), dim=-1).reshape_as(z)
        p = model.model.decode(z, dec_cond)

        latent_permute_loss = -p.log_prob(x_targets).mean()
        loss_dict = {
            'loss': loss_weights['latent_permute_w'] * latent_permute_loss,
            'latent_permute_loss': latent_permute_loss,
        }

        return loss_dict

class CrossEntropyLoss(nn.Module):
    def __init__(self, latent_dim, l_ctx_size, subjects, tasks, norm=False, weight_norm=False):
        super(CrossEntropyLoss, self).__init__()
        self.subject_proj = nn.Linear(latent_dim*l_ctx_size, len(subjects), bias=False)
        self.task_proj = nn.Linear(latent_dim*l_ctx_size, len(tasks), bias=False)
        self.subject_log_temp = nn.Parameter(torch.tensor(0.0))
        self.task_log_temp = nn.Parameter(torch.tensor(0.0))
        self.subject_to_idx = {s: i for i, s in enumerate(subjects)}
        self.task_to_idx = {t: i for i, t in enumerate(tasks)}
        self.norm = norm
        self.weight_norm = weight_norm

    def forward(self, model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights):
        z = z.flatten(1)
        subjects_idx = np.vectorize(self.subject_to_idx.get)(subjects) # Convert subjects to indices
        tasks_idx = np.vectorize(self.task_to_idx.get)(tasks) # Convert tasks to indices
        if self.norm:
            z = F.normalize(z, p=2, dim=-1)
        if self.training:
            if not self.weight_norm:
                subject_logits = self.subject_proj(z) * self.subject_log_temp.exp()
            else:
                subject_logits = F.linear(z, F.normalize(self.subject_proj.weight, p=2, dim=-1)) * self.subject_log_temp.exp()
            subject_loss = F.cross_entropy(subject_logits, torch.tensor(subjects_idx, device=z.device))
        else:
            subject_loss = torch.tensor(0) # subject loss is not available for validation
        if not self.weight_norm:
            task_logits = self.task_proj(z) * self.task_log_temp.exp()
        else:
            task_logits = F.linear(z, F.normalize(self.task_proj.weight, p=2, dim=-1)) * self.task_log_temp.exp()
        task_loss = F.cross_entropy(task_logits, torch.tensor(tasks_idx, device=z.device))
        loss_dict = {
            'loss': loss_weights['ce_w'] * (subject_loss + task_loss),
            'subject_temp': self.subject_log_temp.exp(),
            'task_temp': self.task_log_temp.exp(),
            'subject_loss': subject_loss,
            'task_loss': task_loss,
        }
        return loss_dict

class PairwiseContrastiveLoss(nn.Module):
    def __init__(self, latent_dim, l_ctx_size, norm=True):
        super(PairwiseContrastiveLoss, self).__init__()
        self.channels = latent_dim*l_ctx_size
        self.proj = nn.Linear(self.channels, self.channels, bias=False)
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.norm = norm

    def forward(self, model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights):
        z = z.flatten(1)
        n, b = batch_shape
        z = z.view(n, b, self.channels)
        z = self.proj(z)
        if self.norm:
            z = F.normalize(z, p=2, dim=-1)
        z1, z2 = z.chunk(2, 0)
        z1 = z1.squeeze()
        z2 = z2.squeeze()
        # Loss
        z_sim = z1 @ z2.transpose(-2, -1)
        z_sim = z_sim * self.log_temp.exp()
        targets = torch.arange(b, device=z.device)
        pair_loss = F.cross_entropy(z_sim, targets)

        loss_dict = {
            'loss': loss_weights['contr_w'] * pair_loss,
            'pair_temp': self.log_temp.exp(),
            'pair_loss': pair_loss,
        }
        return loss_dict

class ContrastiveLoss(nn.Module):
    def __init__(self, latent_dim, l_ctx_size, norm=True, proj=True):
        super(ContrastiveLoss, self).__init__()
        self.channels = latent_dim*l_ctx_size
        self.proj = proj
        self.norm = norm
        if self.proj:
            self.subject_proj = nn.Linear(self.channels, self.channels, bias=False)
            self.task_proj = nn.Linear(self.channels, self.channels, bias=False)
        self.subject_log_temp = nn.Parameter(torch.tensor(0.0))
        self.task_log_temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, model, x, x_targets, subjects, tasks, enc_cond, dec_cond, q, z, p, batch_shape, loss_weights):
        z = z.flatten(1)
        n, s, t = batch_shape
        z = z.view(n, s, t, self.channels) # (n, s, t, channels) where n is number of repetitions, s is number of subjects, t is number of tasks
        if self.proj:
            subj_z = self.subject_proj(z)
            task_z = self.task_proj(z)
        else:
            subj_z, task_z = z.chunk(2, -1)
        if self.norm:
            subj_z = F.normalize(subj_z, p=2, dim=-1)
            task_z = F.normalize(task_z, p=2, dim=-1)
        subj_z1, subj_z2 = subj_z.chunk(2, 0)
        task_z1, task_z2 = task_z.chunk(2, 0)

        # Subject loss
        subj_sim = torch.einsum('nitc,njtc->nij', subj_z1, subj_z2) * self.subject_log_temp.exp()
        subj_targets = torch.arange(s, device=z.device).unsqueeze(0).expand(n//2, -1)
        subj_loss = 0.5 * (F.cross_entropy(subj_sim, subj_targets) + F.cross_entropy(subj_sim.transpose(-2, -1), subj_targets))

        # Task loss
        task_sim = torch.einsum('nsic,nsjc->nij', task_z1, task_z2) * self.task_log_temp.exp()
        task_targets = torch.arange(t, device=z.device).unsqueeze(0).expand(n//2, -1)
        task_loss = 0.5 * (F.cross_entropy(task_sim, task_targets) + F.cross_entropy(task_sim.transpose(-2, -1), task_targets))

        loss_dict = {
            'loss': loss_weights['contr_w'] * (subj_loss + task_loss),
            'subject_temp': self.subject_log_temp.exp(),
            'task_temp': self.task_log_temp.exp(),
            'subject_loss': subj_loss,
            'task_loss': task_loss,
        }
        return loss_dict

class NumTrialsEmbedding(nn.Module):
    def __init__(self, num_steps, latent_dim, context_size):
        super(NumTrialsEmbedding, self).__init__()
        self.num_steps = num_steps
        self.latent_dim = latent_dim
        self.context_size = context_size
        self.embedding = nn.Parameter(torch.empty(num_steps, latent_dim, context_size))
        trunc_normal_(self.embedding, std=0.02)

    def forward(self, x, cond):
        return x + self.embedding[cond]

def positional_cosine_embedding(embedding_dim=256, max_len=585):
    t = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(1, embedding_dim+1, dtype=torch.float32).unsqueeze(0)

    denom = torch.pow(max_len, i/embedding_dim)
    embedding = torch.cos(math.pi * t / denom)

    return embedding

class CosineEmbedding(nn.Module):
    def __init__(self, embedding_dim, embedding_length, channels, normalize='standard', reverse_proj=False):
        super(CosineEmbedding, self).__init__()
        embed = positional_cosine_embedding(embedding_dim, embedding_length)
        if normalize == 'standard':
            embed = (embed-embed.mean(dim=1, keepdim=True))/(embed.std(dim=1, keepdim=True)+1e-6)
        elif normalize == 'l2':
            embed = torch.nn.functional.normalize(embed, p=2, dim=1)
        self.register_buffer('embedding', embed)
        self.embedding_scale = nn.Parameter(torch.tensor(0.0))

        self.reverse_proj = reverse_proj
        if reverse_proj:
            self.embed_proj = LinearGLU(embedding_dim, channels)
        else:
            self.in_proj = LinearGLU(channels, embedding_dim)
            self.out_proj = nn.Sequential(LinearGLU(embedding_dim, embedding_dim), nn.Linear(embedding_dim, channels, bias=False))

    def forward(self, x, cond):
        assert x.ndim == cond.ndim + 2
        emb = self.embedding[cond] * self.embedding_scale.exp()
        if self.reverse_proj:
            emb = self.embed_proj(emb)
        else:
            emb = emb + self.in_proj(x.flatten(1))
            emb = self.out_proj(emb)
        x = x + emb.reshape_as(x)
        return x

class ERPUNet(nn.Module):
    def __init__(self, context_size, in_channels, channels, latent_dim, num_layers=4, num_heads=8, activation='glu', std_model='channel', num_samples=256, embedding_type='cosine', embedding_size=256, cosine_norm='standard', cosine_reverse_proj=False, encoder_cond=False, decoder_cond=False, interpolated_residual='none'):
        super(ERPUNet, self).__init__()
        self.kernel_size = 2
        self.context_size = context_size
        self.in_channels = in_channels
        self.channels = channels
        self.num_layers = num_layers
        self.l_ctx_size = context_size // (2**num_layers)
        self.activation = activation
        self.num_samples = num_samples
        self.embedding_type = embedding_type

        # Bootstrap num-trials embedding
        if self.embedding_type == 'cosine':
            self.encoder_cond = CosineEmbedding(embedding_size, num_samples, channels * self.l_ctx_size, normalize=cosine_norm, reverse_proj=cosine_reverse_proj) if encoder_cond else IdentityWithCond()
            self.decoder_cond = CosineEmbedding(embedding_size, num_samples, channels * self.l_ctx_size, normalize=cosine_norm, reverse_proj=cosine_reverse_proj) if decoder_cond else IdentityWithCond()
        elif self.embedding_type == 'disabled':
            self.encoder_cond = IdentityWithCond()
            self.decoder_cond = IdentityWithCond()
        else:
            self.encoder_cond = NumTrialsEmbedding(num_samples, channels, self.l_ctx_size) if encoder_cond else IdentityWithCond()
            self.decoder_cond = NumTrialsEmbedding(num_samples, channels, self.l_ctx_size) if decoder_cond else IdentityWithCond()

        def get_conv_layer():
            return nn.Sequential(
                ConvBlock(channels, channels, channels, activation=activation),
                nn.Conv1d(channels, channels, self.kernel_size, stride=2, padding=0, bias=False),
                ActNorm(channels)
            )
        layers = [get_conv_layer() for _ in range(num_layers)]
        self.encoder_layers = nn.ModuleList(layers)

        self.interpolated_residual = interpolated_residual
        if self.interpolated_residual == 'global_interp':
            self.residual_interpolation = GlobalInterpolation()
        elif self.interpolated_residual == 'local_interp':
            self.residual_interpolation = LocalInterpolation()
        elif self.interpolated_residual == 'residual_interp':
            self.residual_interpolation = ResidualInterpolation()
        else:
            raise ValueError(f"Interpolated residual {self.interpolated_residual} not recognized.")

        # Encoder
        self.encoder_in = ConvBlock(in_channels, channels, channels, activation=activation)

        # Decoder
        def get_conv_layer():
            return nn.Sequential(
                nn.ConvTranspose1d(channels, channels, self.kernel_size, stride=2, padding=0, bias=False),
                ActNorm(channels),
                ConvBlock(channels, channels, channels, activation=activation),
            )
        layers = [get_conv_layer() for _ in range(num_layers)]
        self.decoder_layers = nn.ModuleList(layers)
        self.decoder_mean_std = ConvBlock(channels, channels, 2*in_channels, activation=activation)
        if std_model == 'channel':
            self.decoder_std = nn.Sequential(LinearGLU(context_size, 1), nn.Softplus())
        elif std_model == 'full':
            self.decoder_std = nn.Sequential(ConvBlock(in_channels, in_channels, in_channels, activation=activation), nn.Softplus())

        self.decoder_stddev = nn.Parameter(torch.tensor(math.log(math.exp(0.03)-1)))

        # Bottleneck
        self.encoder_bottleneck = TransformerBottleneck(channels, num_layers, self.l_ctx_size, num_heads)
        self.decoder_bottleneck = TransformerBottleneck(channels, num_layers, self.l_ctx_size, num_heads)


    def forward(self, x, enc_cond, dec_cond):
        skip = []
        x = self.encoder_in(x)
        res = init_res = x
        for layer in self.encoder_layers:
            skip.append(x)
            x = layer(x)
            x, res = self.residual_interpolation(x, init_res, res)
        x = self.encoder_cond(x, enc_cond)
        x = self.encoder_bottleneck(x)
        x = self.decoder_cond(x, dec_cond)
        x = self.decoder_bottleneck(x)
        res = init_res = x
        for layer in self.decoder_layers:
            x = layer(x)
            x = x + skip.pop()
            x, res = self.residual_interpolation(x, init_res, res)
        mean_std = self.decoder_mean_std(x)
        mean, std = mean_std.chunk(2, 1)
        std = self.decoder_std(std)
        p = td.Normal(loc=mean, scale=std + F.softplus(self.decoder_stddev))
        return None, None, p


class IdentityWithCond(nn.Identity):
    def forward(self, x, cond):
        return x

class CSLPAE(nn.Module):
    def __init__(self, context_size, in_channels, channels, latent_dim, num_layers=4, num_heads=8, activation='glu', std_model='channel', num_samples=256, embedding_type='cosine', embedding_size=256, cosine_norm='standard', cosine_reverse_proj=False, encoder_cond=False, decoder_cond=False, interpolated_residual='none'):
        super(CSLPAE, self).__init__()
        self.kernel_size = 2
        self.context_size = context_size
        self.in_channels = in_channels
        self.channels = channels
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.l_ctx_size = context_size // (2**num_layers)
        self.activation = activation
        self.num_samples = num_samples
        self.embedding_type = embedding_type

        # Bootstrap num-trials embedding
        if self.embedding_type == 'cosine':
            self.encoder_cond = CosineEmbedding(embedding_size, num_samples, channels * self.l_ctx_size, normalize=cosine_norm, reverse_proj=cosine_reverse_proj) if encoder_cond else IdentityWithCond()
            self.decoder_cond = CosineEmbedding(embedding_size, num_samples, channels * self.l_ctx_size, normalize=cosine_norm, reverse_proj=cosine_reverse_proj) if decoder_cond else IdentityWithCond()
        elif self.embedding_type == 'disabled':
            self.encoder_cond = IdentityWithCond()
            self.decoder_cond = IdentityWithCond()
        else:
            self.encoder_cond = NumTrialsEmbedding(num_samples, channels, self.l_ctx_size) if encoder_cond else IdentityWithCond()
            self.decoder_cond = NumTrialsEmbedding(num_samples, channels, self.l_ctx_size) if decoder_cond else IdentityWithCond()

        # Encoder
        self.encoder_in = ConvBlock(in_channels, channels, channels, activation=activation)
        self.encoder = StridedConvolutionalEncoder(channels, self.kernel_size, num_layers, activation=activation, interpolated_residual=interpolated_residual)
        self.encoder_bottleneck = TransformerBottleneck(channels, num_layers, self.l_ctx_size, num_heads)
        self.encoder_latent = ConvBlock(channels, channels, latent_dim, activation=activation)

        # Decoder
        self.decoder_in = ConvBlock(latent_dim, channels, channels, activation=activation)
        self.decoder_bottleneck = TransformerBottleneck(channels, num_layers, self.l_ctx_size, num_heads)
        self.decoder = TransposedConvolutionalDecoder(channels, self.kernel_size, num_layers, activation=activation, interpolated_residual=interpolated_residual)
        self.decoder_out = ConvBlock(channels, channels, in_channels, activation=activation)

    def encode(self, x, cond):
        x = self.encoder_in(x)
        z = self.encoder(x)
        z = self.encoder_cond(z, cond)
        z = self.encoder_bottleneck(z)
        z = self.encoder_latent(z)
        return None, z

    def decode(self, z, cond):
        z = self.decoder_in(z)
        z = self.decoder_cond(z, cond)
        z = self.decoder_bottleneck(z)
        x = self.decoder(z)
        mean = self.decoder_out(x)
        p = td.Normal(loc=mean, scale=1.0)
        return p

    def forward(self, x, enc_cond, dec_cond):
        q, z = self.encode(x, enc_cond)
        p = self.decode(z, dec_cond)
        return q, z, p

class ERPAE(nn.Module):
    def __init__(self, context_size, in_channels, channels, latent_dim, num_layers=4, num_heads=8, activation='glu', std_model='channel', num_samples=256, embedding_type='cosine', embedding_size=256, cosine_norm='standard', cosine_reverse_proj=False, encoder_cond=False, decoder_cond=False, interpolated_residual='none'):
        super(ERPAE, self).__init__()
        self.kernel_size = 2
        self.context_size = context_size
        self.in_channels = in_channels
        self.channels = channels
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.l_ctx_size = context_size // (2**num_layers)
        self.activation = activation
        self.num_samples = num_samples
        self.embedding_type = embedding_type

        # Bootstrap num-trials embedding
        if self.embedding_type == 'cosine':
            self.encoder_cond = CosineEmbedding(embedding_size, num_samples, channels * self.l_ctx_size, normalize=cosine_norm, reverse_proj=cosine_reverse_proj) if encoder_cond else IdentityWithCond()
            self.decoder_cond = CosineEmbedding(embedding_size, num_samples, channels * self.l_ctx_size, normalize=cosine_norm, reverse_proj=cosine_reverse_proj) if decoder_cond else IdentityWithCond()
        elif self.embedding_type == 'disabled':
            self.encoder_cond = IdentityWithCond()
            self.decoder_cond = IdentityWithCond()
        else:
            self.encoder_cond = NumTrialsEmbedding(num_samples, channels, self.l_ctx_size) if encoder_cond else IdentityWithCond()
            self.decoder_cond = NumTrialsEmbedding(num_samples, channels, self.l_ctx_size) if decoder_cond else IdentityWithCond()

        # Encoder
        self.encoder_in = ConvBlock(in_channels, channels, channels, activation=activation)
        self.encoder = StridedConvolutionalEncoder(channels, self.kernel_size, num_layers, activation=activation, interpolated_residual=interpolated_residual)
        self.encoder_bottleneck = TransformerBottleneck(channels, num_layers, self.l_ctx_size, num_heads)
        self.encoder_latent = ConvBlock(channels, channels, latent_dim, activation=activation)

        # Decoder
        self.decoder_in = ConvBlock(latent_dim, channels, channels, activation=activation)
        self.decoder_bottleneck = TransformerBottleneck(channels, num_layers, self.l_ctx_size, num_heads)
        self.decoder = TransposedConvolutionalDecoder(channels, self.kernel_size, num_layers, activation=activation, interpolated_residual=interpolated_residual)
        self.decoder_mean_std = ConvBlock(channels, channels, 2*in_channels, activation=activation)
        if std_model == 'channel':
            self.decoder_std = nn.Sequential(LinearGLU(context_size, 1), nn.Softplus())
        elif std_model == 'full':
            self.decoder_std = nn.Sequential(ConvBlock(in_channels, in_channels, in_channels, activation=activation), nn.Softplus())

        self.decoder_stddev = nn.Parameter(torch.tensor(math.log(math.exp(0.03)-1)))

    def encode(self, x, cond):
        x = self.encoder_in(x)
        z = self.encoder(x)
        z = self.encoder_cond(z, cond)
        z = self.encoder_bottleneck(z)
        z = self.encoder_latent(z)
        return None, z

    def decode(self, z, cond):
        z = self.decoder_in(z)
        z = self.decoder_cond(z, cond)
        z = self.decoder_bottleneck(z)
        x = self.decoder(z)
        mean_std = self.decoder_mean_std(x)
        mean, std = mean_std.chunk(2, 1)
        std = self.decoder_std(std)
        p = td.Normal(loc=mean, scale=std + F.softplus(self.decoder_stddev))
        return p

    def forward(self, x, enc_cond, dec_cond):
        q, z = self.encode(x, enc_cond)
        p = self.decode(z, dec_cond)
        return q, z, p