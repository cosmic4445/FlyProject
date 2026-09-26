
import csv
import math

import numpy as np
import scipy.sparse as sp

MOTORS = ("forward", "left", "right", "jump")


INSTINCT_CHANNELS = {
    "forward": [19, 20],  
    "left": [19, 3],     
    "right": [20, 3],
    "jump": [10, 11],    
}


class FlyBrain:
    DT = 1.0     
    TAU_M = 20.0 
    TAU_S = 5.0  
    T_REF = 2.0  
    NOISE = 0.25 

    TAU_ELIG = 400.0  
    CAP_MULT = 5.0  
    MIN_CAP = 0.05     
    def __init__(self, W, sens_chan, motor_idx, sugar_idx, seed=0, plastic=True):
        """
        W          sparse (post x pre) matrix. Entry = membrane jump-current per spike (signed).
        sens_chan  int array, length N. Sensory channel of each neuron, -1 if not sensory.
        motor_idx  dict name -> int array of neuron indices (forward/left/right/jump).
        sugar_idx  int array of sugar-taste neuron indices.
        plastic    if True, reward() actually changes synapse weights (see below).
        """
        csr = sp.csr_matrix(W, dtype=np.float32) 
        self.n = csr.shape[0]
        self.W_csr = csr
        self.indptr = csr.indptr
        self.col = csr.indices                      
        self.data = csr.data                        
        self.row = np.repeat(np.arange(self.n), np.diff(self.indptr)).astype(np.int32) 
        self.plastic = plastic
        self.sign = np.sign(self.data).astype(np.float32)
        self.cap = np.maximum(np.abs(self.data) * self.CAP_MULT, self.MIN_CAP).astype(np.float32)
        self.elig = np.zeros_like(self.data)
        self._elig_decay = math.exp(-self.DT / self.TAU_ELIG)

        self.sens_chan = np.asarray(sens_chan, dtype=np.int32)
        self.sens_mask = self.sens_chan >= 0
        self.n_channels = int(self.sens_chan.max()) + 1 if self.sens_mask.any() else 0
        self.motor_idx = {k: np.asarray(v, dtype=np.int64) for k, v in motor_idx.items()}
        self.sugar_idx = np.asarray(sugar_idx, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        """Clear membrane state for a fresh attempt. Synapse weights (the learning) are NOT reset here -
        that's what makes the fly's improvement persist across attempts rather than starting over each time."""
        self.v = np.zeros(self.n, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refr = np.zeros(self.n, dtype=np.float32)
        self.elig[:] = 0.0  

    def run(self, ms, drive):
        """Simulate `ms` milliseconds with constant external current `drive` (length N).
        Returns spike counts per neuron. Also accumulates eligibility traces (coincidence of a
        synapse's pre- and post-neuron firing) for reward() to use afterwards."""
        counts = np.zeros(self.n, dtype=np.int32)
        k = self.DT / self.TAU_M
        decay = math.exp(-self.DT / self.TAU_S)
        noise_amp = self.NOISE * math.sqrt(k)
        plastic, elig, edecay, col, row = self.plastic, self.elig, self._elig_decay, self.col, self.row
        for _ in range(int(round(ms / self.DT))):
            self.v += k * (-self.v + self.g + drive) + noise_amp * self.rng.standard_normal(self.n, dtype=np.float32)
            self.v[self.refr > 0] = 0.0
            spk = self.v >= 1.0
            self.v[spk] = 0.0
            self.refr[spk] = self.T_REF
            self.refr = np.maximum(self.refr - self.DT, 0.0)
            spk_f = spk.astype(np.float32)
            self.g = self.g * decay + self.W_csr @ spk_f
            if plastic:
                elig *= edecay
                elig += spk_f[col] * spk_f[row]
            counts += spk
        return counts

    def reward(self, r, lr=0.02):
        """Reward-modulated plasticity: nudge every synapse by lr * r * (how eligible it currently is),
        then clip it back within its bounds. Positive r after a synapse's pre/post neurons recently fired
        together strengthens that synapse (whether excitatory or inhibitory); negative r weakens it.
        Weights can't cross zero (Dale's law: an excitatory synapse stays excitatory) or exceed CAP_MULT
        times their original size, which keeps the network from exploding into runaway firing."""
        if not self.plastic or r == 0.0:
            return
        self.data += (lr * r) * self.elig
        pos = self.sign > 0
        neg = ~pos
        np.clip(self.data, 0.0, self.cap, out=self.data, where=pos)
        np.clip(self.data, -self.cap, 0.0, out=self.data, where=neg)



def build_synthetic(n_channels=21, per_channel=20, pool=300, shared=1200, per_motor=30, n_sugar=40, seed=0, innate=True, plastic=True):
    """A random stand-in fly brain with fixed wiring (until you plug in a real connectome).

    sensory (n_channels x per_channel)  ->  4 hidden pools (one per motor group) + a shared recurrent mess  ->  motor.
    Each pool is wired to 4 sensory channels: with innate=True two are 'instinct' channels (goal smell ->
    forward and steering, floor-ahead -> jump, matching the sensor order FlyBrain.server.lua sends) and two
    are random; with innate=False all four are random. Either way the outer loop can't rewire anything,
    it can only tune the gains and biases around the fixed wiring.
    Hidden neurons are 80% excitatory / 20% inhibitory; sugar neurons project broadly into the hidden layer."""
    rng = np.random.default_rng(seed)
    n_sens = n_channels * per_channel
    h0 = n_sens
    n_hidden = len(MOTORS) * pool + shared
    h1 = h0 + n_hidden
    m0 = h1
    m1 = m0 + len(MOTORS) * per_motor
    s0 = m1
    n = s0 + n_sugar

    inhibitory = np.zeros(n, dtype=bool)
    inhibitory[h0:h1] = rng.random(n_hidden) < 0.2

    rows, cols, vals = [], [], []

    def connect(post_ids, pre_ids, k, w_e, w_i):
        post = np.repeat(post_ids, k)
        pre = rng.choice(pre_ids, size=post.size)
        mag = rng.uniform(0.6, 1.4, size=post.size)
        w = np.where(inhibitory[pre], -w_i, w_e) * mag
        rows.append(post); cols.append(pre); vals.append(w)

    all_sens = np.arange(0, n_sens)
    shared_ids = np.arange(h0 + len(MOTORS) * pool, h1)
    hidden_ids = np.arange(h0, h1)
    for p in range(len(MOTORS)):
        pool_ids = np.arange(h0 + p * pool, h0 + (p + 1) * pool)
        designated = INSTINCT_CHANNELS[MOTORS[p]] if innate and n_channels >= 21 else []
        others = [c for c in range(n_channels) if c not in designated]
        picks = rng.choice(others, size=4 - len(designated), replace=False)
        preferred = np.array(list(designated) + [int(c) for c in picks])
        pref_neurons = np.concatenate([np.arange(c * per_channel, (c + 1) * per_channel) for c in preferred])
        connect(pool_ids, pref_neurons, 8, 0.8, 0.8)     
        connect(pool_ids, all_sens, 3, 0.3, 0.3)       
        connect(pool_ids, pool_ids, 6, 0.2, 0.5)        
        connect(np.arange(m0 + p * per_motor, m0 + (p + 1) * per_motor), pool_ids, 40, 0.3, 0.6)
    connect(shared_ids, all_sens, 6, 0.6, 0.6)
    connect(shared_ids, hidden_ids, 15, 0.25, 0.7)
    connect(hidden_ids, s0 + np.arange(n_sugar), 20, 0.6, 0.6)
    for p in range(len(MOTORS)):                        
        connect(np.arange(m0 + p * per_motor, m0 + (p + 1) * per_motor), shared_ids, 6, 0.3, 0.6)

    W = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
    sens_chan = np.full(n, -1, dtype=np.int32)
    sens_chan[:n_sens] = np.repeat(np.arange(n_channels), per_channel)
    motor_idx = {name: np.arange(m0 + i * per_motor, m0 + (i + 1) * per_motor) for i, name in enumerate(MOTORS)}
    return FlyBrain(W, sens_chan, motor_idx, np.arange(s0, n), seed=seed, plastic=plastic)



def save_csv(brain, neurons_path, synapses_path):
    """Write a brain (including any learning that's happened to its weights) in the CSV format load_csv reads."""
    role = {}
    for i in np.nonzero(brain.sens_chan >= 0)[0]:
        role[int(i)] = f"sensory:{int(brain.sens_chan[i])}"
    for name, idx in brain.motor_idx.items():
        for i in idx:
            role[int(i)] = f"motor:{name}"
    for i in brain.sugar_idx:
        role[int(i)] = "sugar"
    with open(neurons_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "role"])
        for i in range(brain.n):
            w.writerow([i, role.get(i, "")])
    with open(synapses_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pre", "post", "weight"])
        for pre, post, wt in zip(brain.col, brain.row, brain.data):
            w.writerow([int(pre), int(post), float(wt)])


def load_csv(neurons_path, synapses_path, weight_scale=1.0, seed=0, plastic=True):
    """Load a connectome from two CSVs.

    neurons.csv   id,role       role is blank, `sensory:<channel>`, `motor:<forward|left|right|jump>` or `sugar`
    synapses.csv  pre,post,weight   weight is signed (negative = inhibitory), in units of
                                    'threshold fractions per spike'. Use weight_scale to convert.
    Ids can be anything (FlyWire root ids are fine); they are remapped to 0..N-1.
    """
    ids, roles = [], []
    with open(neurons_path, newline="") as f:
        for row in csv.DictReader(f):
            ids.append(row["id"])
            roles.append((row.get("role") or "").strip())
    index = {nid: i for i, nid in enumerate(ids)}
    n = len(ids)

    pre, post, wt = [], [], []
    with open(synapses_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["pre"] in index and row["post"] in index:
                pre.append(index[row["pre"]]); post.append(index[row["post"]]); wt.append(float(row["weight"]))
    W = sp.coo_matrix((np.array(wt, dtype=np.float32) * weight_scale, (post, pre)), shape=(n, n)).tocsr()

    sens_chan = np.full(n, -1, dtype=np.int32)
    motor = {name: [] for name in MOTORS}
    sugar = []
    for i, r in enumerate(roles):
        if r.startswith("sensory:"):
            sens_chan[i] = int(r.split(":", 1)[1])
        elif r.startswith("motor:"):
            name = r.split(":", 1)[1]
            if name not in motor:
                raise ValueError(f"unknown motor role {r!r}; expected one of {MOTORS}")
            motor[name].append(i)
        elif r == "sugar":
            sugar.append(i)
    for name, idx in motor.items():
        if not idx:
            raise ValueError(f"no neurons with role motor:{name} in {neurons_path}")
    if not sugar:
        raise ValueError(f"no neurons with role 'sugar' in {neurons_path}")
    return FlyBrain(W, sens_chan, motor, sugar, seed=seed, plastic=plastic)
