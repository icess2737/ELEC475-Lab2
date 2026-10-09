"""Sequential, restartable Lab 2 experiment queue. Run from the Lab2 folder."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent

def experiments():
    ladder = [('lenet', 'lenet', 'none', False, 5),
              ('lenet_modern', 'lenet_modern', 'none', False, 5),
              ('alexnet_none', 'alexnet', 'none', False, 10),
              ('alexnet_lrn', 'alexnet', 'lrn', False, 10),
              ('alexnet_bn', 'alexnet', 'bn', False, 10),
              ('alexnet_with_aug', 'alexnet', 'bn', True, 15),
              ('zoo_alexnet', 'zoo_alexnet', 'none', False, 10)]
    runs = []
    for dataset in ['cifar10', 'tiny_imagenet']:
        for name, model, norm, aug, epochs in ladder:
            runs.append(dict(name=f'{dataset}_{name}_seed42', model=model, norm=norm,
                             dataset=dataset, augment=aug, epochs=epochs, seed=42))
    for seed in [42, 475, 2035]:
        # Fifteen-epoch controls are separate from the ten-epoch norm ablation.
        runs.append(dict(name=f'cifar10_alexnet_no_aug_seed{seed}', model='alexnet',
                         norm='bn', dataset='cifar10', augment=False, epochs=15, seed=seed))
        for norm, aug, epochs, label in [('none', False, 10, 'none'), ('lrn', False, 10, 'lrn'),
                                        ('bn', False, 10, 'bn'), ('bn', True, 15, 'with_aug')]:
            if seed != 42:
                runs.append(dict(name=f'cifar10_alexnet_{label}_seed{seed}', model='alexnet',
                                 norm=norm, dataset='cifar10', augment=aug, epochs=epochs, seed=seed))
    for seed in [42, 475, 2035]:
        runs.append(dict(name=f'cifar10_alexnet_no_dropout_seed{seed}', model='alexnet', norm='bn',
                         dataset='cifar10', augment=False, epochs=15, seed=seed, dropout=0.0))
    return runs

def main():
    os.chdir(ROOT)
    logs = ROOT / 'logs' / 'postlab'
    logs.mkdir(parents=True, exist_ok=True)
    runs = experiments()
    (logs / 'experiment_plan.json').write_text(json.dumps(runs, indent=2))
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1',
                       MPLBACKEND='Agg', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    commands = []
    for i, run in enumerate(runs):
        name = run['name']
        history = logs / f'{name}.csv'
        result = history.with_suffix('.json')
        command = [sys.executable, '-u', 'train.py', '--model', run['model'], '--norm', run['norm'],
                   '--dataset', run['dataset'], '--epochs', str(run['epochs']), '--seed', str(run['seed']),
                   '--batch-size', '128', '--num-workers', '2', '--threads', '4',
                   '--save-model', f'checkpoints/postlab/{name}.pth',
                   '--plot-file', f'plots/postlab/{name}.png', '--history-file', str(history)]
        if not run['augment']:
            command.append('--no-augment')
        if 'dropout' in run:
            command.extend(['--dropout', str(run['dropout'])])
        commands.append(subprocess.list2cmdline(['python'] + command[1:]))
        (ROOT / 'commands_postlab.txt').write_text('\n'.join(commands), encoding='utf-8')
        if result.exists():
            existing = json.loads(result.read_text())
            if existing['final']['epoch'] == run['epochs']:
                continue
        status = dict(state='running', current=name, index=i+1, total=len(runs), updated=time.time())
        (logs / 'progress.json').write_text(json.dumps(status, indent=2))
        print(f'[{i+1}/{len(runs)}] {name}', flush=True)
        with (logs / f'{name}.txt').open('w', encoding='utf-8') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        if completed.returncode:
            status.update(state='failed', exit_code=completed.returncode)
            (logs / 'progress.json').write_text(json.dumps(status, indent=2))
            raise RuntimeError(f'{name} failed; inspect its log')
    (logs / 'progress.json').write_text(json.dumps(dict(state='training_complete', total=len(runs)), indent=2))
    print('All queued training experiments completed.', flush=True)

if __name__ == '__main__':
    main()
