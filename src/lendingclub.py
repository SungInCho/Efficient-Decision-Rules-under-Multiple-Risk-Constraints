"""
LendingClub experiments of the working paper (Section 5.4 and Appendix D). Uses data/processed/pool_preds.parquet and
hold_preds.parquet, which prepare_lendingclub.py and train_models.py produce (splits.py with the hold set).

Design. In every split the 2015 loans are split at random into U (20%, new applicants, features only), cal (40%) and
test (40%). The plug-in linear program on U (no outcomes) gives the weight theta_hat and its two prices, coverage and
severity, for every level.
  random split   : calibration on cal (2015, same period as the test loans)
  temporal split : calibration on hold (2014 loans never used in training)
Rules (all calibrated by multi-risk CRC on their own score):
  dual      h = (theta_hat c_hat / B + 1 - theta_hat) p_hat, theta_hat from U, all calibration loans set the cut-off
  lab       weight chosen on past labels: grid theta (step 0.05) on one half of the calibration loans, cut-off on the
            other half
  theta0    probability only, p_hat
  theta1    expected loss only, p_hat c_hat
  separate  coverage cut-off on p_hat and severity cut-off on p_hat c_hat, pass only below both

Commands (repo root)                               output in results/lendingclub/
  python src/lendingclub.py table --n 100          lc_table1.csv, lc_table1_levels.csv, lc_table1_runs.csv,
                                                   lc_gains.csv, fig_lc_levels.png, fig_lc_splits.png. Table 1 at the
                                                   main level, positive prices and share accepted per level, risks of
                                                   every split
  python src/lendingclub.py figs                   redraw fig_lc_levels, fig_lc_splits and fig_lc_labels from the
                                                   saved csv files
  python src/lendingclub.py labels --n 50          lc_labels.csv, lc_labels_curve.csv, fig_lc_labels.png. Past
                                                   outcomes m = 1,000 to the whole calibration part
  Every command takes --main (default 40): the main level in $ for Table 1 and fig_lc_splits.
  python src/lendingclub.py check                  plug-in weight and positive prices against linprog on 50,000 loans
Run times: table about 10 minutes, labels about 5 minutes.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

from lc_data import B1, C_HAT, split_pool3

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RES = ROOT / "results" / "lendingclub"
A_COV = 0.10
GRID_LEVELS = np.round(np.exp(np.linspace(np.log(5), np.log(100), 20)), 4).tolist()
MAIN = 40.0                                   # main level of Table 1 and fig_lc_splits; --main changes it
LEVELS = sorted(set(GRID_LEVELS + [MAIN]))
GRID = [round(t, 2) for t in np.arange(0, 1.0001, 0.05)]
RULES = ["dual", "lab", "theta0", "theta1", "separate"]
LABELS = {"dual": r"plug-in dual weight $\hat\theta$", "lab": r"chosen on past labels $\hat\theta_{lab}$",
          "theta0": r"probability only, $\theta = 0$", "theta1": r"expected loss only, $\theta = 1$",
          "separate": "separate cut-offs"}
SHORT = {"dual": "plug-in", "lab": "past labels", "theta0": "θ = 0", "theta1": "θ = 1", "separate": "separate"}
COLORS = {"dual": "#0b5cad", "lab": "#8a5cb8", "theta0": "#7a7a7a", "theta1": "#c4661f", "separate": "#3a9a5b"}
INK, MUTED, GRIDC, BAND = "#1f2328", "#6e7781", "#d0d7de", "#e8eef6"
BUDGETS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000]


# ------------------------------------------------------------------ plug-in dual weight (no labels)
def dual_theta(p, c, a_cov, a_usd):
    """theta_hat and the multipliers from the plug-in linear program (no labels used)."""
    n = len(p)
    res = linprog(-np.ones(n), A_ub=np.vstack([p, p * c]), b_ub=[a_cov * p.sum(), n * a_usd],
                  bounds=(0, 1), method="highs-ipm")
    if res.status != 0:
        raise RuntimeError(res.message)
    lam = -res.ineqlin.marginals
    if lam.sum() <= 0:                       # both prices zero: every case can pass, theta is irrelevant
        return 0.0, lam
    return float(lam[1] * B1 / (lam[0] + lam[1] * B1)), lam


# ------------------------------------------------------------------ data
def load(name):
    df = pd.read_parquet(DATA / name)
    p = df["p_hat"].to_numpy(float)
    c = np.clip(np.nan_to_num(df[C_HAT].to_numpy(float)), 0, B1)
    y = df["y"].to_numpy(int)
    loss = np.where(y == 1, np.nan_to_num(df["c1_true"].to_numpy(float)), 0.0)
    return df, p, c, y, loss


def h(theta, p, c):
    return (theta * c / B1 + 1 - theta) * p


# ------------------------------------------------------------------ conformal risk control
def crc_cut(score, loss, alpha, lbar):
    """sup{t : (sum_i loss_i 1{score_i < t} + lbar) / (n + 1) <= alpha}. The rule passes s < t.
    alpha may be an array (one cut-off per level)."""
    n = len(score)
    o = np.argsort(score, kind="stable")
    s = score[o]
    cum = np.concatenate([[0.0], np.cumsum(loss[o])])
    b = np.atleast_1d(alpha) * (n + 1) - lbar
    k = np.searchsorted(cum, b + 1e-9, side="right") - 1          # cases passed on the calibration sample
    t = np.where(k < 0, -np.inf, np.where(k >= n, np.inf, s[np.clip(k, 0, n - 1)]))
    return t if np.ndim(alpha) else float(t[0])


def cut_cov(score, y, alpha):
    """Coverage cut-off: CRC with loss 1 on the adverse calibration cases only."""
    d = score[y == 1]
    return crc_cut(d, np.ones(len(d)), alpha, 1.0)


def cut_sev(score, loss, alphas):
    return crc_cut(score, loss, alphas, B1)


# ------------------------------------------------------------------ plug-in linear program on U (no labels)
def plugin_prices(p, c, a_cov, levels):
    """Which prices of the plug-in linear program are positive, for every level (no labels).

    With two constraints a price is positive exactly when the rule that ignores its constraint breaks it:
      severity price > 0  <=>  the coverage-only rule (cut-off on p_hat) uses more than the severity level
      coverage price > 0  <=>  the severity-only rule (cut-off on p_hat c_hat) uses more than the coverage level
    (compare with the multipliers of linprog in the check command). Returns two boolean arrays (coverage, severity)."""
    n = len(p)
    pc = p * c
    a1 = a_cov * p.sum() + 1e-12
    a2 = n * np.asarray(levels, float) + 1e-9
    o = np.argsort(p, kind="stable")
    kc = int(np.searchsorted(np.cumsum(p[o]), a1, side="right"))
    sev_of_cov_rule = pc[o][:kc].sum()
    o = np.argsort(pc, kind="stable")
    cs, cp_ = np.concatenate([[0.0], np.cumsum(pc[o])]), np.concatenate([[0.0], np.cumsum(p[o])])
    ks = np.searchsorted(cs, a2, side="right") - 1                 # cases passed by the severity-only rule
    return cp_[ks] > a1, sev_of_cov_rule > a2


def plugin_theta(p, c, a_cov, levels, coarse=0.01, fine=0.001):
    """theta_hat for every level from the plug-in linear program on unlabeled cases (no labels).

    The linear program is solved through its threshold form: for each weight, pass the cases with the lowest
    h_theta while both plug-in constraints hold, and keep the weight that passes the most. By linear-programming
    duality this weight equals lambda_2 B / (lambda_1 + lambda_2 B) from the dual variables, up to the grid step
    (compare with dual_theta in the check command). If the best weights include 0 or 1, that end is returned
    (one constraint does not bind). Otherwise the centre of the best block on a grid of step `fine`.
    About a hundred times faster than linprog on 50,000 cases for all levels at once."""
    n = len(p)
    pc = p * c
    a1 = a_cov * p.sum() + 1e-12
    a2 = n * np.asarray(levels, float) + 1e-9

    def passed(ths):
        out = np.empty((len(ths), len(a2)), int)
        for i, th in enumerate(ths):
            o = np.argsort(h(th, p, c), kind="stable")
            kc = np.searchsorted(np.cumsum(p[o]), a1, side="right")
            ks = np.searchsorted(np.cumsum(pc[o]), a2, side="right")
            out[i] = np.minimum(kc, ks)
        return out

    grid = np.round(np.arange(0, 1 + coarse / 2, coarse), 6)
    K = passed(grid)
    theta = np.empty(len(a2))
    for j in range(len(a2)):
        best = np.flatnonzero(K[:, j] == K[:, j].max())
        if best[0] == 0:
            theta[j] = 0.0
        elif best[-1] == len(grid) - 1:
            theta[j] = 1.0
        else:
            lo, hi = max(0.0, grid[best[0]] - coarse), min(1.0, grid[best[-1]] + coarse)
            g2 = np.round(np.arange(lo, hi + fine / 2, fine), 6)
            k2 = passed(g2)[:, j]
            b2 = g2[k2 == k2.max()]
            theta[j] = float((b2[0] + b2[-1]) / 2)
    return theta


def labeled_theta(p, c, y, loss, alphas):
    """Grid weight (step 0.05) that accepts the most of these labeled cases after calibrating on them, per level."""
    share = np.empty((len(GRID), len(alphas)))
    for i, th in enumerate(GRID):
        s = h(th, p, c)
        t = np.minimum(cut_cov(s, y, A_COV), cut_sev(s, loss, alphas))
        share[i] = (s[:, None] < t[None, :]).mean(axis=0)
    return np.array(GRID)[np.argmax(share, axis=0)]


def metrics(passed, y, loss):
    return dict(accept=100 * passed.mean(), miss=(passed & (y == 1)).sum() / y.sum(), harm=(passed * loss).sum() / len(y))


def calibrated(score_cal, y, loss, alphas):
    """Multi-risk CRC cut-off per level on one score."""
    return np.minimum(cut_cov(score_cal, y, A_COV), cut_sev(score_cal, loss, alphas))


def rules_on(cal, test, th_dual, alphas, rng):
    """Cut-offs of the five rules from one calibration sample, evaluated on the test loans. Returns rows per level."""
    p1, c1, y1, l1 = cal
    p2, c2, y2, l2 = test
    rows = []
    # plug-in dual weight: one weight per level, all calibration loans
    for j, th in enumerate(th_dual):
        s1 = h(th, p1, c1)
        t = min(cut_cov(s1, y1, A_COV), cut_sev(s1, l1, alphas[j]))
        rows.append(("dual", j, th, metrics(h(th, p2, c2) < t, y2, l2)))
    # weight chosen on past labels: half A picks theta, half B calibrates
    perm = rng.permutation(len(p1))
    A, Bh = perm[: len(p1) // 2], perm[len(p1) // 2:]
    th_lab = labeled_theta(p1[A], c1[A], y1[A], l1[A], alphas)
    for j, th in enumerate(th_lab):
        s1 = h(th, p1[Bh], c1[Bh])
        t = min(cut_cov(s1, y1[Bh], A_COV), cut_sev(s1, l1[Bh], alphas[j]))
        rows.append(("lab", j, th, metrics(h(th, p2, c2) < t, y2, l2)))
    # theta = 0 and theta = 1
    for rule, th in [("theta0", 0.0), ("theta1", 1.0)]:
        s1, s2 = h(th, p1, c1), h(th, p2, c2)
        t = calibrated(s1, y1, l1, alphas)
        for j in range(len(alphas)):
            rows.append((rule, j, th, metrics(s2 < t[j], y2, l2)))
    # separate cut-offs: coverage on p_hat, severity on p_hat c_hat, accept only below both
    tc = cut_cov(p1, y1, A_COV)
    ts = cut_sev(p1 * c1, l1, alphas)
    for j in range(len(alphas)):
        rows.append(("separate", j, np.nan, metrics((p2 < tc) & (p2 * c2 < ts[j]), y2, l2)))
    return rows


def summarize_new(runs, keys, n_splits):
    """Means over splits, Monte Carlo SE, share of each level used, and the share of splits above each level."""
    runs = runs.assign(exc_cov=runs.miss > A_COV, exc_sev=runs.harm > runs.alpha_sev)
    g = runs.groupby(keys)
    s = g[["accept", "miss", "harm", "theta"]].mean()
    s = s.join((g[["accept", "miss", "harm"]].std() / np.sqrt(n_splits)).add_prefix("se_"))
    s = s.join(100 * g[["exc_cov", "exc_sev"]].mean()).reset_index()
    s["freq_used"] = 100 * s.miss / A_COV
    s["loss_used"] = 100 * s.harm / s.alpha_sev
    s["z_cov"] = (s.miss - A_COV) / s.se_miss.clip(lower=1e-12)
    s["z_sev"] = (s.harm - s.alpha_sev) / s.se_harm.clip(lower=1e-12)
    return s


def gains(summ, runs):
    """Per setting and level: share of splits with two positive prices, and the gain in share passed of the plug-in
    dual weight over the better single score (theta = 0 or 1, better in mean), over separate cut-offs and over the
    weight chosen on past labels (pp)."""
    two = runs[runs.rule == "dual"].groupby(["setting", "alpha_sev"]).n_pos.apply(lambda v: (v == 2).mean())
    acc = summ.pivot_table(index=["setting", "alpha_sev"], columns="rule", values="accept")
    out = pd.DataFrame({"share_two_pos": two,
                        "accept_dual": acc["dual"],
                        "gain_vs_single": acc["dual"] - np.maximum(acc["theta0"], acc["theta1"]),
                        "gain_vs_separate": acc["dual"] - acc["separate"],
                        "gain_vs_lab": acc["dual"] - acc["lab"]})
    return out.reset_index()


def table(n_splits):
    """Table 1 at the main level, positive prices and gains per level, risks of every split. Random and temporal split."""
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    _, pp, cp, yp, lp = load("pool_preds.parquet")
    _, ph, ch, yh, lh = load("hold_preds.parquet")
    alphas = np.array(LEVELS)
    print(f"2015 loans {len(pp):,} (U 20%, cal 40%, test 40%), hold loans {len(ph):,}", flush=True)

    recs = []
    for seed in range(n_splits):
        rng = np.random.default_rng(20_000 + seed)
        U, C, T = split_pool3(len(pp), seed)
        th_dual = plugin_theta(pp[U], cp[U], A_COV, alphas)
        pos_cov, pos_sev = plugin_prices(pp[U], cp[U], A_COV, alphas)
        n_pos = pos_cov.astype(int) + pos_sev.astype(int)
        test = (pp[T], cp[T], yp[T], lp[T])
        for setting, cal in [("random", (pp[C], cp[C], yp[C], lp[C])), ("temporal", (ph, ch, yh, lh))]:
            for rule, j, th, m in rules_on(cal, test, th_dual, alphas, rng):
                recs.append(dict(seed=seed, setting=setting, rule=rule, alpha_sev=LEVELS[j], theta=th,
                                 n_pos=n_pos[j], pos_cov=pos_cov[j], pos_sev=pos_sev[j], **m))
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  {seed + 1}/{n_splits} splits  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    runs.to_csv(RES / "lc_table1_runs.csv", index=False)
    summ = summarize_new(runs, ["setting", "rule", "alpha_sev"], n_splits)
    summ.to_csv(RES / "lc_table1_levels.csv", index=False)
    main = summ[np.isclose(summ.alpha_sev, MAIN)]
    tab = main.pivot(index="rule", columns="setting", values=["accept", "freq_used", "loss_used"]).loc[RULES]
    tab = tab.reorder_levels([1, 0], axis=1).sort_index(axis=1, level=0)[
        [(s, v) for s in ("random", "temporal") for v in ("accept", "freq_used", "loss_used")]]
    tab.to_csv(RES / "lc_table1.csv")

    pd.set_option("display.width", 220, "display.max_rows", 500)
    print(f"\nTable 1 at alpha_cov = {A_COV}, alpha_sev = ${MAIN:g} (means over {n_splits} splits; "
          f"used = risk / level x 100):")
    print(tab.rename(index=SHORT).round(1).to_string())
    print(f"Largest SE of the share accepted at ${MAIN:g}: {main.se_accept.max():.2f} pp")
    print(f"\nWeights at ${MAIN:g} (mean over splits): " + ", ".join(
        f"{s} {r} {main[(main.setting == s) & (main.rule == r)].theta.iloc[0]:.3f}"
        for s in ("random", "temporal") for r in ("dual", "lab")))

    # positive prices of the plug-in linear program on U, per level
    pr = runs[(runs.rule == "dual") & (runs.setting == "random")].groupby("alpha_sev")
    pos = pd.DataFrame({"two positive (%)": 100 * pr.n_pos.apply(lambda v: (v == 2).mean()),
                        "coverage only (%)": 100 * pr.apply(lambda d: (d.pos_cov & ~d.pos_sev).mean()),
                        "severity only (%)": 100 * pr.apply(lambda d: (~d.pos_cov & d.pos_sev).mean()),
                        "theta_hat": pr.theta.mean()})
    print("\nPositive prices of the plug-in linear program on U, share of splits per level:")
    print(pos.round(1).to_string())

    gn = gains(summ, runs)
    gn.to_csv(RES / "lc_gains.csv", index=False)
    print("\nGain of the plug-in dual weight in share accepted (pp), per level:")
    print(gn.pivot(index="alpha_sev", columns="setting",
                   values=["share_two_pos", "gain_vs_single", "gain_vs_separate", "gain_vs_lab"]).round(2).to_string())
    print("\nMean gain over the better single score, by levels with two positive prices in most splits or not:")
    print(gn.assign(two=gn.share_two_pos > 0.5).groupby(["setting", "two"])[
        ["gain_vs_single", "gain_vs_separate", "gain_vs_lab"]].agg(["mean", "min", "max"]).round(2).to_string())

    print("\nGuarantees over all levels. exceeded = mean above the level by more than 2 standard errors; "
          "split share = % of splits above the level (largest over levels, and at the main level):")
    agg = summ.groupby(["setting", "rule"]).agg(
        levels_cov_exceeded=("z_cov", lambda z: int((z > 2).sum())),
        levels_sev_exceeded=("z_sev", lambda z: int((z > 2).sum())),
        max_freq_used=("freq_used", "max"), max_loss_used=("loss_used", "max"),
        max_split_share_cov=("exc_cov", "max"), max_split_share_sev=("exc_sev", "max"))
    at60 = main.set_index(["setting", "rule"])[["exc_cov", "exc_sev"]].add_prefix("split_share_main_")
    print(agg.join(at60).round(1).to_string())
    print("(CRC bounds the mean risk over splits; single splits above the level are expected in the random split)")

    figures(summ, runs)
    print(f"\nSaved lc_table1.csv, lc_table1_levels.csv, lc_table1_runs.csv, lc_gains.csv, fig_lc_levels.png and "
          f"fig_lc_splits.png to {RES}  [{time.time() - t0:.0f}s]")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRIDC, lw=0.6)
    ax.tick_params(colors=INK, labelsize=8)


def figures(summ, runs):
    """fig_lc_levels: random split, share accepted by each rule against the level, with the levels where both
    plug-in prices are positive in most splits shaded. fig_lc_splits: risk of every split at the main level."""
    two = runs[(runs.rule == "dual") & (runs.setting == "random")].groupby("alpha_sev").n_pos.apply(
        lambda v: (v == 2).mean())
    two = two.index[two > 0.5]
    acc = summ[summ.setting == "random"].pivot(index="alpha_sev", columns="rule", values="accept")
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    if len(two):
        ax.axvspan(two.min(), two.max(), color=BAND, lw=0, zorder=0, label="both prices positive")
    for r in RULES:
        ax.plot(acc.index, acc[r], color=COLORS[r], marker="o", ms=2.5, lw=1.8 if r == "dual" else 1.1,
                zorder=3 if r == "dual" else 2, label=LABELS[r])
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 50, 100])
    ax.set_xticklabels(["5", "10", "20", "50", "100"])
    ax.minorticks_off()
    ax.set_xlabel(r"severity level $\alpha_{\mathrm{sev}}$ (\$ per applicant)", fontsize=8.5, color=INK)
    ax.set_ylabel("share accepted (%)", fontsize=8.5, color=INK)
    style(ax)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RES / "fig_lc_levels.png", dpi=300)

    # risk of every split at the main level. Boxes: quartiles; whiskers: 5th and 95th percentiles; dot: mean
    d = runs[np.isclose(runs.alpha_sev, MAIN)]
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.4), sharex=True, sharey="row")
    for col, setting in enumerate(("random", "temporal")):
        for row, (v, lab) in enumerate([("freq", "coverage used (%)"), ("loss", "severity used (%)")]):
            ax = axes[row, col]
            x = d[d.setting == setting]
            vals = [100 * (x[x.rule == r].miss / A_COV if v == "freq" else x[x.rule == r].harm / MAIN) for r in RULES]
            bp = ax.boxplot(vals, widths=0.6, patch_artist=True, showfliers=False, whis=(5, 95), showmeans=True,
                            meanprops=dict(marker="o", markersize=3.5, markerfacecolor=INK, markeredgecolor=INK))
            for patch, r in zip(bp["boxes"], RULES):
                patch.set_facecolor(COLORS[r])
                patch.set_alpha(0.35)
            for med in bp["medians"]:
                med.set_color(INK)
            ax.axhline(100, color=MUTED, lw=0.8, ls="--")
            if col == 0:
                ax.set_ylabel(lab, fontsize=8.5, color=INK)
            if row == 0:
                ax.set_title(f"{setting} split", fontsize=8.5, color=INK)
            style(ax)
            ax.grid(False)
        axes[1, col].set_xticks(range(1, len(RULES) + 1))
        axes[1, col].set_xticklabels([SHORT[r] for r in RULES], rotation=30, ha="right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(RES / "fig_lc_splits.png", dpi=300)


def labels_figure(out):
    """fig_lc_labels from lc_labels.csv: share accepted at the main level against the number of labeled loans m.
    Shaded: m < 2 B / alpha_sev, where the m/2 loans left for calibration are too few for the severity cut-off to
    pass anyone (CRC needs (sum of losses + B) / (n + 1) <= alpha_sev, so n + 1 > B / alpha_sev)."""
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    lim = 2 * B1 / MAIN
    if out.index.min() < lim:
        ax.axvspan(out.index.min() / 1.15, lim, color="#eeeeee", lw=0, zorder=0)
        ax.text(np.sqrt(out.index.min() / 1.15 * lim), out[["dual_main", "lab_main"]].to_numpy().max() * 0.97,
                "half the m\ntoo few to\ncalibrate", ha="center", va="top", fontsize=6.5, color=MUTED,
                bbox=dict(facecolor="#eeeeee", edgecolor="none", pad=1))
    for col, rule, lab in [("dual_main", "dual", "plug-in prices: all m loans calibrate"),
                           ("lab_main", "lab", "weight chosen on m/2 loans, m/2 calibrate")]:
        ax.plot(out.index, out[col], color=COLORS[rule], marker="o", ms=3, lw=1.4, label=lab, zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(out.index.min() / 1.15, out.index.max() * 1.15)
    ax.set_xlabel("labeled loans m", fontsize=8.5, color=INK)
    ax.set_ylabel(f"share accepted at ${MAIN:g} (%)", fontsize=8.5, color=INK)
    style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(RES / "fig_lc_labels.png", dpi=300)


def labels(n_splits):
    """Appendix table. Random split. m past outcomes drawn from cal: plug-in dual weight (all m calibrate) against the
    weight chosen on the labels (m/2 choose, m/2 calibrate). Printed at the main level and averaged over the levels where both
    plug-in prices are positive in most splits."""
    t0 = time.time()
    RES.mkdir(parents=True, exist_ok=True)
    _, pp, cp, yp, lp = load("pool_preds.parquet")
    alphas = np.array(LEVELS)

    recs = []
    for seed in range(n_splits):
        rng = np.random.default_rng(30_000 + seed)
        U, C, T = split_pool3(len(pp), seed)
        th_dual = plugin_theta(pp[U], cp[U], A_COV, alphas)
        pos_cov, pos_sev = plugin_prices(pp[U], cp[U], A_COV, alphas)
        p2, c2, y2, l2 = pp[T], cp[T], yp[T], lp[T]
        order = rng.permutation(C)
        for m in [b for b in BUDGETS if b < len(C)] + [len(C)]:
            L = order[:m]
            A, Bh = L[: m // 2], L[m // 2:]
            th_lab = labeled_theta(pp[A], cp[A], yp[A], lp[A], alphas)
            for j, a in enumerate(alphas):
                for rule, th, cal in [("dual", th_dual[j], L), ("lab", th_lab[j], Bh)]:
                    s1 = h(th, pp[cal], cp[cal])
                    t = min(cut_cov(s1, yp[cal], A_COV), cut_sev(s1, lp[cal], a))
                    recs.append(dict(seed=seed, m=m, rule=rule, alpha_sev=LEVELS[j], theta=th,
                                     both=bool(pos_cov[j] and pos_sev[j]), **metrics(h(th, p2, c2) < t, y2, l2)))
        if (seed + 1) % 10 == 0 or seed == 0:
            print(f"  {seed + 1}/{n_splits} splits  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    w = runs.pivot_table(index=["seed", "m", "alpha_sev"], columns="rule", values="accept")
    d = (w["dual"] - w["lab"]).rename("diff").reset_index()
    both = runs[runs.rule == "dual"].groupby("alpha_sev").both.mean()
    mid = both.index[both > 0.5]
    acc = runs.groupby(["m", "alpha_sev", "rule"]).accept.mean().unstack("rule")
    out = pd.DataFrame({
        "dual_main": acc.xs(MAIN, level="alpha_sev")["dual"],
        "lab_main": acc.xs(MAIN, level="alpha_sev")["lab"],
        "diff_main": d[np.isclose(d.alpha_sev, MAIN)].groupby("m")["diff"].mean(),
        "se_diff_main": d[np.isclose(d.alpha_sev, MAIN)].groupby("m")["diff"].std() / np.sqrt(n_splits),
        "diff_mean_two_pos": d[d.alpha_sev.isin(mid)].groupby("m")["diff"].mean(),
        "lab_accepts_none_main": runs[(runs.rule == "lab") & np.isclose(runs.alpha_sev, MAIN)]
        .groupby("m").accept.apply(lambda a: 100 * (a == 0).mean()),
    })
    out.to_csv(RES / "lc_labels.csv")
    pd.set_option("display.width", 220)
    print(f"\nPast outcomes m (random split). Share accepted (%) at ${MAIN:g}, plug-in dual minus past labels (pp), "
          f"and the mean difference over the {len(mid)} levels where both plug-in prices are positive:")
    print(out.round(2).to_string())
    print(f"lab_accepts_none_main: % of splits in which the weight chosen on past labels accepts no one at ${MAIN:g}")
    curve = runs[runs.alpha_sev.isin(mid)].groupby(["m", "rule"]).accept.mean().rename("accept").reset_index()
    curve.to_csv(RES / "lc_labels_curve.csv", index=False)
    labels_figure(out)
    print(f"\nSaved lc_labels.csv, lc_labels_curve.csv and fig_lc_labels.png to {RES}  [{time.time() - t0:.0f}s]")


def check(n_cases=50_000):
    """plugin_theta and plugin_prices against linprog (dual_theta) on one sample of 2015 loans, at a few levels."""
    _, pp, cp, _, _ = load("pool_preds.parquet")
    idx = np.random.default_rng(1).choice(len(pp), min(n_cases, len(pp)), replace=False)
    lv = [5.0, 10.0, 25.0, 40.0, MAIN, 80.0, 100.0]
    fast = plugin_theta(pp[idx], cp[idx], A_COV, lv)
    pos_cov, pos_sev = plugin_prices(pp[idx], cp[idx], A_COV, lv)
    for a, f, pc_, ps_ in zip(lv, fast, pos_cov, pos_sev):
        th, lam = dual_theta(pp[idx], cp[idx], A_COV, a)
        print(f"  alpha_sev ${a:>5.1f}: theta linprog {th:.4f}, fast {f:.4f};  positive prices linprog "
              f"({lam[0] > 1e-9}, {lam[1] > 1e-9}), fast ({bool(pc_)}, {bool(ps_)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="LendingClub experiments (Section 5.4, Appendix D)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("table", "labels", "figs"):
        sp = sub.add_parser(name)
        if name != "figs":
            sp.add_argument("--n", type=int, default=50 if name == "labels" else 100, help="number of random splits")
        sp.add_argument("--main", type=float, default=40.0, help="main level in $ (Table 1, fig_lc_splits)")
    sub.add_parser("check")
    args = ap.parse_args()
    if hasattr(args, "main"):
        MAIN = args.main
        LEVELS = sorted(set(GRID_LEVELS + [MAIN]))
    if args.cmd == "table":
        table(args.n)
    elif args.cmd == "labels":
        labels(args.n)
    elif args.cmd == "check":
        check()
    else:
        figures(pd.read_csv(RES / "lc_table1_levels.csv"), pd.read_csv(RES / "lc_table1_runs.csv"))
        print(f"Redrew fig_lc_levels.png and fig_lc_splits.png in {RES}")
        if (RES / "lc_labels.csv").exists():
            labels_figure(pd.read_csv(RES / "lc_labels.csv", index_col="m"))
            print(f"Redrew fig_lc_labels.png in {RES} from lc_labels.csv (main level of that run)")