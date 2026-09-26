
import json
import os

import numpy as np

from brain import MOTORS

AMP_IN = 3.0        
AMP_BIAS = 1.2      
BIAS = {"forward": 1.5, "left": 0.0, "right": 0.0, "jump": -0.3}

WINDOW_MS = 150      
SUBFRAMES = 3      


FWD_LO, FWD_SPAN = 10.0, 40.0
TURN_SPAN = 50.0
JUMP_THRESHOLD = 40.0


DX_PER_STEP = 2.5    
LR_STEP = 0.03         
LR_TERMINAL = 0.15     
DEATH_REWARD = -1.0
CLEAR_REWARD_BASE = 1.5
CLEAR_REWARD_PER_LEVEL = 0.15 


class Fly:
    def __init__(self, brain, state_path=None):
        self.brain = brain
        self.C = brain.n_channels
        self.state_path = state_path
        self.weights_path = (os.path.splitext(state_path)[0] + "_weights.npz") if state_path else None

        self.bias = np.array([BIAS[m] for m in MOTORS], dtype=np.float32)
        self.attempt = 0
        self.deaths = 0
        self.clears = 0
        self.best_progress = 0.0
        self.last_progress = 0.0
        self._load()

   
    def start_attempt(self):
        self.attempt += 1
        self.brain.reset()
        return self._stats()

    def end_attempt(self, progress, level, cleared):
        if cleared:
            self.clears += 1
            r = CLEAR_REWARD_BASE + CLEAR_REWARD_PER_LEVEL * level
        else:
            self.deaths += 1
            r = DEATH_REWARD
        self.brain.reward(r, lr=LR_TERMINAL)
        self.best_progress = max(self.best_progress, progress)
        self.last_progress = progress
        if self.attempt % 10 == 0:
            self._save()
        return self._stats()

 
    def _drive(self, sensors):
        b = self.brain
        x = np.zeros(self.C, dtype=np.float32)
        m = min(len(sensors), self.C)
        x[:m] = np.clip(np.asarray(sensors[:m], dtype=np.float32), -1.5, 1.5)
        drive = np.zeros(b.n, dtype=np.float32)
        drive[b.sens_mask] = AMP_IN * x[b.sens_chan[b.sens_mask]]
        for i, name in enumerate(MOTORS):
            drive[b.motor_idx[name]] += AMP_BIAS * self.bias[i]
        return drive

    @staticmethod
    def _decode(rates):
        f = min(max((rates["forward"] - FWD_LO) / FWD_SPAN, 0.0), 1.0)
        t = min(max((rates["right"] - rates["left"]) / TURN_SPAN, -1.0), 1.0)
        return {"f": round(f, 3), "t": round(t, 3), "j": rates["jump"] > JUMP_THRESHOLD}

    def step(self, sensors, dx=0.0):
        """One brain tick. `dx` is how far forward (in studs) the fly moved since the *previous*
        /step call - i.e. the result of the actions we handed back last time - so we reward
        those just-executed actions before computing the next ones."""
        b = self.brain
        r = float(np.clip(dx / DX_PER_STEP, -1.0, 1.0))
        b.reward(r, lr=LR_STEP)

        drive = self._drive(sensors)
        sub_ms = WINDOW_MS / SUBFRAMES
        actions = []
        last_rates = {}
        for _ in range(SUBFRAMES):
            counts = b.run(sub_ms, drive)
            rates = {n: float(counts[b.motor_idx[n]].sum()) / (len(b.motor_idx[n]) * sub_ms / 1000.0) for n in MOTORS}
            actions.append(self._decode(rates))
            last_rates = rates
        return {"actions": actions, "window": WINDOW_MS / 1000.0, "stats": self._stats(), "rates": last_rates}

    def taste(self, level, tier):
        """Fire the sugar-taste neurons - this is flavor/feedback (and it's what a real fly's brain
        would light up on eating something sweet), separate from the numeric reward() call in
        end_attempt that actually does the learning."""
        b = self.brain
        drive = np.zeros(b.n, dtype=np.float32)
        drive[b.sugar_idx] = 2.0 + 0.6 * tier
        sugar = 0
        for _ in range(3):
            counts = b.run(100, drive)
            sugar += int(counts[b.sugar_idx].sum())
        return {"sugar_spikes": sugar, "stats": self._stats()}


    def _stats(self):
        return {
            "neurons": int(self.brain.n),
            "synapses": int(self.brain.W_csr.nnz),
            "attempt": self.attempt,
            "deaths": self.deaths,
            "clears": self.clears,
            "best": round(self.best_progress, 3),
            "last": round(self.last_progress, 3),
        }

    def _save(self):
        if not self.state_path:
            return
        data = {
            "attempt": self.attempt, "deaths": self.deaths, "clears": self.clears,
            "best_progress": self.best_progress,
        }
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.state_path)
        if self.weights_path:
            tmpw = self.weights_path + ".tmp.npz"
            np.savez_compressed(tmpw, data=self.brain.data)
            os.replace(tmpw, self.weights_path)

    def _load(self):
        if self.state_path and os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    d = json.load(f)
                self.attempt, self.deaths, self.clears = d["attempt"], d["deaths"], d["clears"]
                self.best_progress = d["best_progress"]
                print(f"resumed fly: attempt {self.attempt}, best progress {self.best_progress:.2f}")
            except (OSError, ValueError, KeyError) as e:
                print("couldn't read state file, starting fresh:", e)
        if self.weights_path and os.path.exists(self.weights_path):
            try:
                saved = np.load(self.weights_path)["data"]
                if saved.shape == self.brain.data.shape:
                    self.brain.data[:] = saved
                    print("resumed learned synapse weights")
                else:
                    print("saved weights don't match this brain's shape, starting fresh")
            except (OSError, ValueError, KeyError) as e:
                print("couldn't read weights file, starting fresh:", e)
