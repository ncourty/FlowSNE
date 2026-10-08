# GPU benchmark results

Produced by `benchmarks/gpu_benchmark.py` (see its docstring for what is
measured). Three runs so far, same machine.

## Run 1: RTX 2080 Ti, 9 Oct 2026

- **Machine:** host `dmis`, NVIDIA GeForce RTX 2080 Ti (11.55 GB), Linux 6.1
  (Debian 12), Python 3.9.19, torch 2.2.2+cu121, TorchDR 0.4. pykeops 2.2.3
  as first installed; KeOps' engine initially compiled against an older CUDA
  toolkit's `nvrtc.h` (errors about `nvrtcGetCUBIN`) and was fixed by giving
  it CUDA-12 headers before this run.
- **Settings:** script defaults: float32, perplexity 50 (k = 250
  neighbours), row-stochastic input affinity, sphere vortex embedded in 50
  dimensions, 3 warm-up + 10 timed steps per configuration.
- The raw JSON was not added to the repository; numbers below are from the
  terminal output.

**Correctness on the GPU** (n = 2000, dense vs KeOps)

| dtype | kNN agreement | loss rel. diff | grad Z rel. diff | grad θ rel. diff |
|---|---|---|---|---|
| float32 | 1.0000 | 0 | 5.5e-7 | 1.2e-7 |
| float64 | 1.0000 | 1.1e-16 | 4.1e-15 | 2.6e-15 |

**Scaling.** "Per step" = loss + backward + Adam step, median of 10. Dense runs
are skipped when they would need more than 80% of GPU memory.

| n | input affinity (KeOps kNN) | KeOps per step | KeOps peak memory | dense per step | dense peak memory |
|---|---|---|---|---|---|
| 10,000 | 1.7 s | 0.020 s | 0.26 GB | 0.097 s | 3.86 GB |
| 20,000 | 2.4 s | 0.034 s | 0.49 GB | skipped (~10 GB) | – |
| 50,000 | 8.1 s | 0.112 s | 1.21 GB | skipped (~60 GB) | – |
| 100,000 | 19.9 s | 0.359 s | 1.73 GB | skipped (~240 GB) | – |

**End-to-end fit** (n = 50,000, 1000 iterations, KeOps): 124 s in total, loss
17.87, corr(‖fitted field‖, true speed) = +0.79.

### Reading

- **KeOps vs dense on the same GPU:** 5× faster at 10k, with 15× less memory.
  Against the dense path on a laptop CPU (Apple M4: 7.1 s/step at 10k), about
  360× faster.
- **Memory grows linearly in n** (1.7 GB at 100k): time, not memory, is now
  the limit. A 3000-iteration fit at 100k is about 18 min on this card.
- **Time per step grows more slowly than n² so far** (×3.2 from 50k to 100k,
  against ×4 for pure n²): fixed overheads and the O(nk) attraction term are
  still visible. Extrapolating, 10⁶ points would be about 100× slower per
  step, impractical without approximating the repulsion term (e.g. negative
  sampling).
- **Quality at scale was not established by this run** (answered by run 2
  below: +0.94 with 3000 iterations and perplexity √n). The 50k fit's speed
  correlation (+0.79) is below the n = 1500 value (+0.89, 3000 iterations,
  perplexity √n), but this run used 1000 iterations and perplexity 50, so it
  is not a like-for-like comparison. To check: `--fit-iters 3000`, and
  `--perplexity auto`.

## Run 2: same machine, quality at scale, 9 Oct 2026

`python gpu_benchmark.py --sizes 50000 --fit-n 50000 --fit-iters 3000 --perplexity auto`
(run with the script as of run 1, i.e. before the KeOps-on-GPU check was
added; correctness results identical to run 1).

With `--perplexity auto` (√n ≈ 224, k = 1119 neighbours):

| n = 50,000 | perplexity 50, k = 250 (run 1) | auto, k = 1119 (run 2) |
|---|---|---|
| input affinity | 8.1 s (peak 0.74 GB) | 109.7 s (peak 2.12 GB) |
| KeOps per step | 0.112 s (peak 1.21 GB) | 0.227 s (peak 3.19 GB) |
| end-to-end fit | 1000 iterations: 124 s, loss 17.87, corr +0.79 | 3000 iterations: 802 s, loss 17.58, **corr +0.94** |

### Reading

- **Quality holds at scale:** with 3000 iterations and FlowSNE's default
  perplexity, the fitted field's speed correlation is +0.94 at 50k, above the
  +0.89 obtained at n = 1500.
- **Confounded:** iterations and perplexity changed together, so the gain
  over run 1 cannot be attributed from this run alone (separated by run 3).
- **Cost of a large k:** KeOps' kNN search (`argKmin`) is the step that
  suffers, 14× slower at k = 1119 than at k = 250; each fit step is 2× slower
  (the O(nk) attraction term) and memory 2.6× higher.

## Run 3: separating iterations from perplexity, 9 Oct 2026

`python benchmarks/gpu_benchmark.py --sizes 50000 --fit-n 50000 --fit-iters 3000 --perplexity 50`
(script with the KeOps-on-GPU check, which passed with a relative error below
1e-7; correctness and the n = 50k timings are identical to run 1: input
affinity 8.08 s, 0.1106 s/step, peak 1.21 GB).

End-to-end fit, n = 50,000, KeOps:

| | 1000 iterations | 3000 iterations |
|---|---|---|
| perplexity 50 (k = 250) | corr +0.79 (run 1, 124 s) | corr **+0.82** (this run, 356 s, loss 17.17) |
| perplexity √n ≈ 224 (k = 1119) | – | corr **+0.94** (run 2, 802 s, loss 17.58) |

### Reading

- **Perplexity is what matters:** tripling the iterations adds +0.03; moving
  to the √n perplexity adds +0.12. A wider neighbourhood is what lets the
  flow recover the drift's magnitude at this scale.
- **Do not cap the perplexity at 50 to save time:** it halves the fit's cost
  (356 s vs 802 s) but gives up most of the quality. The √n default is
  justified here; an intermediate value (e.g. 100) is untested.
- Losses are not comparable across runs with different perplexities (each
  has its own input affinity P).
