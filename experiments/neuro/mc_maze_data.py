"""MC_Maze (Churchland et al. 2012; Neural Latents Benchmark, DANDI 000128):
macaque M1/PMd spiking during delayed reaches through a maze. Builds the
condition-averaged population trajectories that jPCA uses, and their
velocities, from the NWB file (read with h5py, no pynwb needed).

Pipeline (standard jPCA preprocessing): successful trials; spikes binned at
1 ms, Gaussian-smoothed (sigma 20 ms), aligned to movement onset, window
[-250, +450] ms sampled every 10 ms; averaged per condition (trial_type,
trial_version); soft-normalised per neuron, rate / (range + 5 Hz). Saved
raw (not mean-subtracted), so the cross-condition mean can be removed or not
downstream.

    conda run -n dev python experiments/neuro/mc_maze_data.py --file full
Writes experiments/neuro/data/mc_maze_{file}.npz:
  R (C, T, N) soft-normalised rates, t (T,) ms from movement onset,
  cond (C, 2), n_trials (C,), hand_vel (C, T, 2), hand_pos (C, T, 2)
"""
import argparse
import os

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter1d

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
T0, T1, STEP, SIGMA = -250, 450, 10, 20          # ms


def ragged(f, name):
    d, idx = f[name][:], f[name + "_index"][:]
    return np.split(d, idx[:-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="full", choices=["small", "full"])
    a = ap.parse_args()
    f = h5py.File(os.path.join(DATA, f"mc_maze_{a.file}.nwb"), "r")
    tr = f["intervals/trials"]
    ok = tr["success"][:] & (tr["split"][:].astype(str) != "test")
    ok &= ~np.isnan(tr["move_onset_time"][:])
    onset = tr["move_onset_time"][:][ok]
    cond = np.stack([tr["trial_type"][:][ok], tr["trial_version"][:][ok]], 1)
    spikes = ragged(f, "units/spike_times")
    N = len(spikes)
    pad = 4 * SIGMA
    edges = np.arange(T0 - pad, T1 + pad + 1) / 1000.0     # 1 ms bins
    keep = np.arange(pad, pad + (T1 - T0) + 1, STEP)        # sample times
    t = np.arange(T0, T1 + 1, STEP)
    rates = np.zeros((len(onset), len(t), N), np.float32)
    for u, s in enumerate(spikes):
        for k, o in enumerate(onset):
            lo, hi = np.searchsorted(s, [o + edges[0], o + edges[-1]])
            h = np.histogram(s[lo:hi] - o, edges)[0] * 1000.0
            rates[k, :, u] = gaussian_filter1d(h.astype(np.float32), SIGMA)[keep]
    beh = {}
    for key in ("hand_vel", "hand_pos"):
        ts = f[f"processing/behavior/{key}/timestamps"][:]
        dat = f[f"processing/behavior/{key}/data"][:]
        idx = np.searchsorted(ts, onset[:, None] + t[None] / 1000.0).clip(0, len(ts) - 1)
        beh[key] = dat[idx]                                   # (trials, T, 2)
    keys, inv = np.unique(cond, axis=0, return_inverse=True)
    inv = inv.ravel()
    C = len(keys)
    mean = lambda A: np.stack([A[inv == c].mean(0) for c in range(C)])  # noqa: E731
    R = mean(rates)
    ntr = np.bincount(inv, minlength=C)
    rng = R.max((0, 1)) - R.min((0, 1))
    R = R / (rng + 5.0)
    out = os.path.join(DATA, f"mc_maze_{a.file}.npz")
    np.savez_compressed(out, R=R, t=t, cond=keys, n_trials=ntr,
                        hand_vel=mean(beh["hand_vel"]), hand_pos=mean(beh["hand_pos"]))
    print(f"{len(onset)} trials, {C} conditions (trials/cond {ntr.min()}-{ntr.max()}), "
          f"{N} units, {len(t)} time points -> {out}")


if __name__ == "__main__":
    main()
