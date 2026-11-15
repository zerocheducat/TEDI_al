"""Actual auxiliary budget protocol; preview by default, 20 periodic/random runs."""
import argparse
import csv
import subprocess
import sys
from pathlib import Path
from tedi.matched_budget import TARGETS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--device',default='cuda',choices=['cpu','cuda','auto'])
    p.add_argument('--envs',nargs='+',default=list(TARGETS),choices=list(TARGETS))
    p.add_argument('--modes',nargs='+',default=['periodic','random'],choices=['stagnation','periodic','random'])
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--output_dir',default='./TEDI_runs/matched_aux_20261115')
    p.add_argument('--cost_csv',default=None,help='Optional original T03 CSV; verify every target against paper Full rows')
    a=p.parse_args()
    if a.cost_csv:
        with open(a.cost_csv,encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        for env in a.envs:
            for seed,target in TARGETS[env].items():
                matches=[r for r in rows if r['environment']==env and int(r['seed'])==seed and r['method']=='Full TEDI' and r['experiment_group']=='paper_main']
                if len(matches)!=1 or int(matches[0]['auxiliary_steps_total'])!=target:
                    raise ValueError(f'CSV mismatch: {env} seed {seed}')
        print('Original Full auxiliary targets verified against CSV',flush=True)
    for env in a.envs:
        seeds=a.seeds or list(TARGETS[env])
        if any(s not in TARGETS[env] for s in seeds):
            p.error(f'unsupported paper seed for {env}')
        for seed in seeds:
            for mode in a.modes:
                cmd=[sys.executable,'main.py','--env',env,'--seed',str(seed),'--device',a.device,
                     '--matched_aux_budget','--trigger_mode',mode,'--eval_actor','actor' if env=='Ant-v2' else 'ema',
                     '--steps','1000000','--pop_size','6','--output_dir',a.output_dir,
                     '--run_name',f'{env}_{mode}_auxmatched_s{seed}','--no_overwrite']
                print(subprocess.list2cmdline(cmd),flush=True)
                if a.execute:
                    subprocess.run(cmd,cwd=Path(__file__).resolve().parent.parent,check=True)


if __name__=='__main__':main()
