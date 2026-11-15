"""Export per-run measured costs; missing/incomplete runs remain visible."""
import argparse
import csv
import json
from pathlib import Path
from tedi.matched_budget import TARGETS


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('root',nargs='?',default='./TEDI_runs/matched_aux_20261115')
    p.add_argument('--modes',nargs='+',default=['periodic','random'],choices=['stagnation','periodic','random'])
    a=p.parse_args();root=Path(a.root)
    rows=[]
    for env,seeds in TARGETS.items():
        for seed,target in seeds.items():
            for mode in a.modes:
                path=root/env/f'{env}_{mode}_auxmatched_s{seed}'/'training_summary.json'
                row=dict(environment=env,seed=seed,method=mode,aux_target=target,status='missing')
                if path.exists():
                    d=json.loads(path.read_text(encoding='utf-8'))
                    row.update({k:d.get(k) for k in ('main_steps','aux_actual','probe_steps','parent_steps','calls','admitted_transitions','unused_auxiliary_budget','auxiliary_overshoot','budget_exhausted','peak','independent_final_mean','raw_return_AUC','wall_time_seconds')})
                    lengths=d.get('hc_full_call_costs',[])
                    actual=d.get('aux_actual',-1)
                    passed=(target<=actual<=target+5 if env=='Ant-v2' else actual==target and d.get('calls')==target//7000 and len(lengths)==target//7000 and all(x==7000 for x in lengths))
                    passed=passed and d.get('main_steps')==1000000 and d.get('aux_target')==target
                    row['status']='matched' if passed else 'budget_not_matched'
                rows.append(row)
    root.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    out=root/'budget_audit.csv'
    with out.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    print(out)
    print({s:sum(r['status']==s for r in rows) for s in ('matched','budget_not_matched','missing')})


if __name__=='__main__':main()
