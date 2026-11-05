"""Figures for article/ICML from experiments/theory/results."""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "..", "..", "article", "ICML", "figures")
os.makedirs(FIG, exist_ok=True)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
C = {"linear_global": "#2a78d6", "squared_global": "#eb6834", "linear_perrow": "#1baf7a"}
LABEL = {"linear_global": "linear cost, global bandwidth",
         "squared_global": "squared cost, global bandwidth",
         "linear_perrow": "linear cost, per-row bandwidth"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "serif"})


def fields_figure(npz):
    d = np.load(npz)
    X, p = d["X"], d["p"]
    rho = float(d["rho"])
    panels = [(-3 * rho * d["gphi"], r"gradient part $-3\rho\nabla\phi$"),
              (-3 * rho * d["w"], r"solenoidal part $-3\rho\,\Pi_p\omega$"),
              (d["brs"], "row-stochastic transport"),
              (d["bds"], "balanced transport")]
    fig, axs = plt.subplots(1, 4, figsize=(6.75, 1.95), constrained_layout=True)
    G = 11
    cell = np.minimum((X * G).astype(int), G - 1)
    lin = cell[:, 0] * G + cell[:, 1]
    cnt = np.bincount(lin, minlength=G * G)
    cx, cy = np.meshgrid((np.arange(G) + 0.5) / G, (np.arange(G) + 0.5) / G, indexing="ij")

    def binned(V):
        return np.stack([np.bincount(lin, V[:, k], G * G) / np.maximum(cnt, 1) for k in (0, 1)], 1)
    B = [binned(V) for V, _ in panels]
    scale = max(np.linalg.norm(b, axis=1).max() for b in B) * G * 0.75
    for ax, (V, title), b in zip(axs, panels, B):
        ax.scatter(X[:, 0], X[:, 1], s=1.0, c=p, cmap="Greys", vmin=0, vmax=p.max() * 2.2,
                   linewidths=0, rasterized=True)
        ax.quiver(cx.ravel(), cy.ravel(), b[:, 0], b[:, 1], angles="xy", scale_units="xy",
                  scale=scale, width=0.007, headwidth=3, headlength=3.5, headaxislength=3, color=INK, pivot="mid")
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color(GRID)
    fig.savefig(os.path.join(FIG, "torus_fields.pdf"), dpi=300)
    plt.close(fig)


def leakage_figure(res):
    leak = res["torus"]["leakage"]
    fig, axs = plt.subplots(1, 2, figsize=(3.3, 2.15), sharey=True, constrained_layout=True)
    for ax, kind, title in [(axs[0], "grad", "gradient drift"), (axs[1], "rot", "solenoidal drift")]:
        for cost in ["linear_global", "squared_global", "linear_perrow"]:
            v = leak[f"{kind}_{cost}"]
            y = np.array(v["transport_rms"])
            if kind == "grad" and cost == "linear_global":
                continue
            ax.plot(v["rho"], y, "-o", color=C[cost], lw=1.5, ms=3, mec="#fcfcfb", mew=1,
                    label=LABEL[cost])
        r = np.array(leak["grad_linear_global"]["rho"])
        ref = leak[f"{kind}_linear_perrow"]["transport_rms"][0]
        ax.plot(r, ref * (r / r[0]) * 0.35, ":", color=INK2, lw=1)
        ax.text(r[-1], ref * (r[-1] / r[0]) * 0.35, r"  $\propto\rho$", color=INK2, va="center", fontsize=7)
        if kind == "grad":
            ref3 = leak["grad_squared_global"]["transport_rms"][-1]
            ax.plot(r, ref3 * (r / r[-1]) ** 3, "--", color=INK2, lw=1)
            ax.text(r[1], ref3 * (r[1] / r[-1]) ** 3, r"$\propto\rho^3$  ", color=INK2, ha="right", fontsize=7)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"drift strength $\rho$", fontsize=7)
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, which="major", color=GRID, lw=0.5)
    axs[0].set_ylabel("RMS transport (units of $h$)", fontsize=7)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6.5, loc="outside lower center", ncol=1)
    fig.savefig(os.path.join(FIG, "leakage.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(RES, "gauge_checks_n3000_*.json")))[-1]
    res = json.load(open(path))
    leakage_figure(res)
    fields_figure(os.path.join(RES, "torus_fields.npz"))
    print("figures written to", FIG)
