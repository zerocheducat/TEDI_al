import unittest
from types import SimpleNamespace
import numpy as np
import torch
from tedi.matched_budget import MatchedBudget,TARGETS
from tedi.evolution import TEDIEvolutionMixin
from tedi.config import EnvironmentConfig


class BudgetTests(unittest.TestCase):
    def test_hc_schedule(self):
        for seed,target in TARGETS['HalfCheetah-v2'].items():
            for mode in ('periodic','random'):
                b=MatchedBudget('HalfCheetah-v2',seed,mode)
                self.assertEqual(len(b.schedule),target//7000)
                self.assertTrue(all(x%5000==0 and 180000<=x<=1000000 for x in b.schedule))
                self.assertTrue(all(y-x>=75000 for x,y in zip(b.schedule,b.schedule[1:])))
                self.assertEqual(b.schedule,MatchedBudget('HalfCheetah-v2',seed,mode).schedule)

    def test_fifo_guard_and_exhaustion(self):
        b=MatchedBudget('HalfCheetah-v2',10,'periodic')
        x=SimpleNamespace(disable_evolution=False,disable_trigger=False,matched_budget=b,
            shadow_env_steps=0,evolution_count=0,config=EnvironmentConfig.create('HalfCheetah-v2'),
            last_evolution_step=-10**18,stagnation_patience=0)
        def can(step,reward):
            return TEDIEvolutionMixin._can_evolve(x,step,reward,999,reward,0,999)
        due=b.schedule[0]
        self.assertFalse(can(due,0));self.assertEqual(b.cursor,0)
        self.assertTrue(can(due+5000,6000))
        b.called(due+5000);x.last_evolution_step=due+5000
        self.assertFalse(can(due+10000,6000))
        x.shadow_env_steps=b.target
        self.assertFalse(can(1000000,6000))
        self.assertFalse(b.report(b.target-1,9,0)['budget_match_pass'])
        b.hc_costs=[7000]*9
        self.assertTrue(b.report(b.target,9,9000)['budget_match_pass'])

    def test_ant_real_rollout_atomic_stop_and_masks(self):
        class Env:
            def reset(self,seed):return np.zeros(2,dtype=np.float32),{}
            def step(self,action):return np.zeros(2,dtype=np.float32),1.,False,False,{}
        class Fake(TEDIEvolutionMixin):
            def _mutate_policy(self,parent,sigma,role):return parent
            def _write_text(self,msg):pass
            def _write_event(self,event):self.event=event
        b=MatchedBudget('Ant-v2',15,'periodic');b.target=7
        actor=torch.nn.Linear(2,1)
        x=Fake();x.trigger_mode='periodic';x.matched_budget=b;x.config=EnvironmentConfig.create('Ant-v2');x.pop_size=6
        x.seed=15;x.evolution_count=0;x.shadow_env_steps=0;x.parent_shadow_steps=0;x.max_action=1.
        x.shadow_pool=SimpleNamespace(reset=lambda seed:np.zeros((6,2),dtype=np.float32),envs=[Env() for _ in range(6)])
        x.td3=SimpleNamespace(actor=actor,select_action=lambda state,actor:np.zeros(1,dtype=np.float32),
                              critic=lambda s,a:(torch.ones(len(s),1),torch.ones(len(s),1)))
        transitions=[]
        x.buffer=SimpleNamespace(size=0,add=lambda *t:transitions.append(t))
        x.device=torch.device('cpu');x.log_candidates=False;x.disable_filtering=False;x.unlimited_injection=False
        x.inject_rng=np.random.default_rng(37)
        x.run_evolution(40000,0.)
        self.assertEqual(x.shadow_env_steps,12)
        self.assertEqual(x.event['probe_lengths'],[2]*6)
        self.assertEqual(x.event['collection_stop_reason'],'auxiliary_budget')
        self.assertTrue(transitions)
        self.assertTrue(all(t[-1]==0 for t in transitions))
        self.assertTrue(b.report(12,1,0)['budget_match_pass'])
        x.run_evolution(80000,0.)
        self.assertEqual(x.evolution_count,1)

        # Exercise the original parent rollout as well: shared auxiliary counter
        # includes two parent steps once, plus 6*2 probe steps. No HC top-up.
        hc=MatchedBudget('HalfCheetah-v2',15,'periodic');hc.target=7000
        x.matched_budget=hc;x.config=EnvironmentConfig.create('HalfCheetah-v2')
        x.config.values['evolution_horizon']=2
        x.evolution_count=0;x.shadow_env_steps=0;x.parent_shadow_steps=0
        x.parent_shadow_env=Env();x.td3.ema_actor=actor
        x.td3.critic_target=x.td3.critic
        x.run_evolution(300000,6000.)
        self.assertEqual(x.shadow_env_steps,14)
        self.assertEqual(x.parent_shadow_steps,2)
        self.assertEqual(hc.hc_costs,[14])
        self.assertFalse(hc.report(14,1,2)['budget_match_pass'])
        x.run_evolution(400000,6000.)
        self.assertEqual(x.evolution_count,1)


if __name__=='__main__':unittest.main()
