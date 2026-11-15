"""20261115 protocol: actual auxiliary steps, including parent reference steps."""
import math
import random

TARGETS = {
    'Ant-v2': {15:44111,50:22486,20:31692,30:16136,55:11538},
    'HalfCheetah-v2': {15:42000,10:63000,25:70000,40:70000,55:56000},
}


class MatchedBudget:
    def __init__(self, env, seed, mode):
        self.env, self.mode = env, mode
        self.target = TARGETS[env][seed]
        self.cursor = 0
        self.actual_call_steps = []
        self.boundary_lengths = []
        self.hc_costs = []
        rng = random.Random(seed + 91773)
        if mode == 'stagnation':
            self.schedule = []
        elif env == 'HalfCheetah-v2':
            k = self.target // 7000
            if mode == 'periodic':
                self.schedule = [math.ceil((180000 + (i+.5)*820000/k)/5000)*5000 for i in range(k)]
            else:
                # Uniform sample among K-subsets of the 5K grid with >=75K gaps.
                # Compress out 14 grid cells between neighbours, then restore.
                grid = list(range(180000,1000001,5000))
                compressed = sorted(rng.sample(range(len(grid)-14*(k-1)), k))
                self.schedule = [grid[x+14*i] for i,x in enumerate(compressed)]
        else:
            self.schedule = []
            step = 0
            while True:
                step += 40000 if mode == 'periodic' else rng.randint(20000,60000)
                if step > 1000000:
                    break
                self.schedule.append(step)

    def available(self, actual, calls):
        return actual < self.target and (self.env != 'HalfCheetah-v2' or calls < self.target//7000)

    def requested(self, step):
        return self.mode == 'stagnation' or (self.cursor < len(self.schedule) and step >= self.schedule[self.cursor])

    def called(self, step):
        self.actual_call_steps.append(step)
        if self.mode != 'stagnation':
            self.cursor += 1

    def report(self, actual, calls, parent_steps):
        matched = self.target <= actual <= self.target+5 if self.env == 'Ant-v2' else (
            actual == self.target and calls == self.target//7000 and len(self.hc_costs) == calls and all(c==7000 for c in self.hc_costs))
        return dict(aux_target=self.target, aux_actual=actual, probe_steps=actual-parent_steps,
                    parent_steps=parent_steps, calls=calls, planned_schedule=self.schedule,
                    actual_call_steps=self.actual_call_steps, budget_exhausted=actual>=self.target,
                    unused_auxiliary_budget=max(0,self.target-actual), auxiliary_overshoot=max(0,actual-self.target),
                    budget_match_pass=matched, budget_boundary_probe_lengths=self.boundary_lengths,
                    hc_full_call_costs=self.hc_costs)
