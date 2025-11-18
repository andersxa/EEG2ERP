import wandb
import argparse
from tqdm import tqdm
from eeg2erp.evaluation import test_model
from eeg2erp.data import ERPDataset, ERPBootstrapTargets, ERPCoreNormalizer
import os


class Baseline:
    def __init__(self, name):
        self.name = 'base_'+name
        self.group = "X"+name[0].upper()
        self.id = None
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('--path', type=str, default='data/')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--compile', type=int, default=2)
    parser.add_argument("--groups", type=str, default='') #D6,D4,D3,C,U1,U2,B4,A3,B3,A4,B2,B1,A2,A
    parser.add_argument('--models', type=str, default='')
    parser.add_argument("--print_log", type=int, default=1)
    parser.add_argument("--prepend_id", type=int, default=0)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--difference_waveform_measures', type=int, default=0)
    parser.add_argument('--save_wandb', type=int, default=1)
    
    # Model
    parser.add_argument("--override_target_type", type=str, default="")

    # Baselines
    parser.add_argument('--baselines', type=str, default='') #leonowicz,kotowski,molina_fast,simple

    # Saving and evaluation
    parser.add_argument('--num_points', type=int, default=25)
    parser.add_argument('--trajectory_spacing', type=str, default='fill')
    parser.add_argument('--repeats', type=int, default=200)

    parser.add_argument('--eval_all', type=int, default=0)
    parser.add_argument('--max_step_eval', type=int, default=100)
    parser.add_argument('--override_steps', type=str, default='')
    parser.add_argument('--chronological_sampling', type=int, default=0)

    args = parser.parse_args()
    device = 'cuda'
    print("Loading data...")
    test_dataset = ERPDataset(path=args.path, split=args.split, processing='simple', no_leakage=True, restricted=True).deterministic().to(device, data=True)
    test_bootstrap_targets = ERPBootstrapTargets(args.path, split=args.split, processing='simple', repeats=args.repeats, prominent_channel_only=True, chronological_sampling=bool(args.chronological_sampling)).to(device)
    normalizer = ERPCoreNormalizer(path=args.path, processing='simple')
    
    api = wandb.Api()
    groups = [g.strip() for g in args.groups.split(',') if g.strip()]
    runs = list(api.runs("user/eeg2erp", filters={"$or": [{"group": g} for g in groups]}))
    for baseline in args.baselines.split(','):
        if baseline:
            runs.append(Baseline(baseline.strip().lower()))
    for run in tqdm(runs):
        if args.models and run.name not in args.models.split(','):
            continue
        args.model = run.name
        args.id = run.id
        # Add chronological suffix if enabled
        chrono_suffix = "_chrono" if args.chronological_sampling else ""

        if args.eval_all:
            result_file = f'{args.path}Results/{run.group}/{args.split}/{run.name}_results_all{chrono_suffix}.pt'
        else:
            result_file = f'{args.path}Results/{run.group}/{args.split}/{run.name}_results{chrono_suffix}.pt'
        # Check if file already exists
        print("Checking", result_file)
        if os.path.exists(result_file):
            continue
        print(f"Testing model {run.name} from group {run.group}...")
        try:
            df, results = test_model(args, test_dataset=test_dataset, test_bootstrap_targets=test_bootstrap_targets, normalizer=normalizer)
        except FileNotFoundError as e:
            print(f"Could not find model {run.name}. Skipping...")
            continue
        if args.difference_waveform_measures:
            df = df[0]
        if not args.eval_all:
            print("Mean over all subjects and tasks:")
            mean_over_all = df.groupby(['Subject', 'Task']).mean(numeric_only=True).mean()
            if args.save_wandb:
                run_log = {}
                for column, value in mean_over_all.items():
                    run_log[args.split+'/'+column] = value
                run_log[args.split+'/final_tested'] = True
                if isinstance(run, Baseline):
                    with wandb.init(project='eeg2erp', name=run.name, group=run.group) as current_run:
                        current_run.log(run_log)
                else:
                    with wandb.init(project='eeg2erp', id=run.id, resume="allow") as current_run:
                        current_run.log(run_log)
        