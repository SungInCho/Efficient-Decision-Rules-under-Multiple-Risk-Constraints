"""
Simulation study of the working paper (Section 5 and Appendix C).

Two risks. X ~ N(0, I_3), p(x) = sigmoid(-2.2 + x1 - 0.5 tanh(x2)), c(x) = sigmoid(-1 + 1.5 x2 - 0.3 x1 + 0.5 x3).
  Y ~ Bernoulli(p(x)). Given Y = 1 the loss is C ~ Beta(4c, 4(1 - c)), so that E[C | X, Y = 1] = c. C = 0 if Y = 0.
  Coverage: Pr(pass | Y = 1) <= alpha_cov = 0.10, in linear form E[phi p] <= alpha_cov pi_1.
  Severity: E[C 1{Y = 1, pass}] <= alpha_sev, in linear form E[phi pc] <= alpha_sev.
Three risks (command three). X ~ N(0, I_4). The third risk is late payment: a loan that is paid late at least once but
  is repaid in full. It causes no loss of principal but collection work and cost, so it enters as a frequency.
  A borrower is prone to pay late with probability r(x) = sigmoid(-3 + x4 + 0.4 x1), independent of Y given X, and a
  late payment is counted only if the loan does not default: Y_late = 1{prone} (1 - Y), so
  q(x) = Pr(Y_late = 1 | x) = (1 - p(x)) r(x), about 6% on average (default about 14%).
  Late payment: Pr(Y_late = 1, pass) <= alpha_late, in linear form E[phi q] <= alpha_late. Like p, q is fitted by a
  classifier in the cal and labels parts. Through x4, q is not a function of p and pc.
Positive prices: the plug-in linear program  max E[phi]  s.t.  E[phi Z_k] <= a_k,  0 <= phi <= 1  is solved on cases
  without outcomes (lp_prices). Its prices lambda give the score lambda . Z. A price counts as positive when
  lambda_k E[Z_k] > TOL, the average part of the score that comes from risk k.

Commands (repo root)                                   output in results/simulation/
  python src/simulation.py fig1a                       fig_shape.png. Pass regions of four rules in the (p, pc) plane
  python src/simulation.py levels                      fig_levels.png, fig_levels.csv. Two risks, true scores:
                                                       positive prices per level, gain of the combined score over the
                                                       single scores and separate cut-offs, share of each level used
  python src/simulation.py three --parts map           three_map.csv, fig_three_map.png. Three risks, true scores, grid
                                                       of (alpha_sev, alpha_late): positive prices, combined score against
                                                       the scores that drop one risk, single scores, separate cut-offs
  python src/simulation.py three --parts cal --reps 100
                                                       three_cal_runs.csv, three_cal_summary.csv, fig_three_splits.png.
                                                       Fitted scores, prices from cases without outcomes, multi-risk CRC,
                                                       at cells with three, two and one positive prices
  python src/simulation.py three --parts labels --reps 100
                                                       three_labels_runs.csv, three_labels_summary.csv,
                                                       fig_three_labels.png. Same m labeled cases: plug-in prices
                                                       (all m calibrate) against weights chosen on m/2 labels
  python src/simulation.py errors --reps 100           errors_runs.csv, errors_summary.csv, fig_errors.png. Two risks:
                                                       value of better estimates of p and c (bias in both
                                                       directions and random noise)
  python src/simulation.py fig2                        fig_theorem2.png, fig_theorem2_data.npz. (a) sampling error of
                                                       the weight from cases without outcomes, (b) the full method
  python src/simulation.py fig2 --plot-only            redraw fig2 from the saved data
Run times (two cores): fig1a seconds, levels about 5 minutes, three map about 10 minutes, three cal about 25 minutes
with --reps 100, three labels about 30 minutes with --reps 100, errors about 10 minutes with --reps 100, fig2 about 25 minutes.
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog

OUT = "results/simulation"
A_COV, RATIO, B = 0.10, 0.5, 1.0
TOL = 1e-6                                     # a price is positive when lambda_k E[Z_k] > TOL
INK, MUTED, GRID, CLOUD = "#1f2328", "#6e7781", "#d0d7de", "#c9ccd1"
BLUE, ORANGE, GREEN, DOT = "#0b4f9c", "#c4661f", "#3a9a5b", "#d9622b"
PURPLE, BAND = "#8a5cb8", "#e8eef6"
# same colours as the LendingClub figures: theta = 0 grey, theta = 1 orange, separate cut-offs green, blend blue
COL = {"p": "#5a6068", "pc": "#c4661f", "rect": "#3a9a5b", "opt": "#0b4f9c"}


# ---------------------------------------------------------------- data
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def logit(q):
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return np.log(q / (1 - q))


def draw_x(n, rng, d=3):
    return rng.standard_normal((n, d))


def true_scores(x):
    # no interaction term: dlogit(p)/dx1 = 1 everywhere, so (p, pc) has a continuous density (Assumption A.1)
    p = sigmoid(-2.2 + x[:, 0] - 0.5 * np.tanh(x[:, 1]))
    c = sigmoid(-1.0 + 1.5 * x[:, 1] - 0.3 * x[:, 0] + 0.5 * x[:, 2])
    return p, c


def draw_labels(p, c, rng, k=4.0):
    """Y ~ Bernoulli(p). Given Y = 1, C ~ Beta(k c, k (1 - c)), so that E[C | X, Y = 1] = c. C = 0 if Y = 0."""
    y = (rng.random(len(p)) < p).astype(int)
    cc = np.clip(c, 1e-6, 1 - 1e-6)
    loss = np.where(y == 1, rng.beta(k * cc, k * (1 - cc)), 0.0)
    return y, loss


def late_scores(x):
    """Probability q = (1 - p) r of a late payment on a loan that is repaid (only loans that do not default count)."""
    p, _ = true_scores(x)
    return (1 - p) * sigmoid(-3.0 + x[:, 3] + 0.4 * x[:, 0])


def risks3(x):
    """Expected use of the three constraints per case, as columns: p (coverage), pc (severity), q (late payment)."""
    p, c = true_scores(x)
    return np.column_stack([p, p * c, late_scores(x)])


def labels3(x, rng):
    """Outcomes of the three-risk model: default Y and its loss C, late payment Y_late and its loss, 1 per late loan."""
    p, c = true_scores(x)
    y, loss = draw_labels(p, c, rng)
    prone = (rng.random(len(x)) < sigmoid(-3.0 + x[:, 3] + 0.4 * x[:, 0])).astype(int)
    y2 = prone * (1 - y)                                            # a loan that defaults is not counted as late
    return y, loss, y2, y2.astype(float)


# ---------------------------------------------------------------- plug-in linear program (no labels)
def lp_prices(Z, a):
    """Prices of max E[phi] s.t. E[phi Z_k] <= a_k (k = 1, ..., K), 0 <= phi <= 1, on the rows of Z. The optimal rule
    passes the cases with lambda . z < 1. Returns lambda and the share passed."""
    n = len(Z)
    res = linprog(-np.ones(n), A_ub=Z.T, b_ub=n * np.asarray(a, float), bounds=(0, 1), method="highs-ipm")
    if res.status != 0:
        raise RuntimeError(res.message)
    return np.maximum(-res.ineqlin.marginals, 0.0), -res.fun / n


def score_shares(lam, Z):
    """Average part of the score lambda . Z that comes from each risk, lambda_k E[Z_k] / sum_j lambda_j E[Z_j]."""
    u = lam * Z.mean(axis=0)
    return u / u.sum() if u.sum() > 0 else u


def prefix_count(score, Z, A):
    """Number of cases passed when the cases with the lowest score pass for as long as sum_i Z_ik <= A_k for all k.
    A may hold arrays: A[0] a scalar, A[1] an array of length m1, A[2] an array of length m2 (result of shape
    (m1, m2)), so that one sort serves a whole grid of levels."""
    o = np.argsort(score, kind="stable")
    cum = np.cumsum(Z[o], axis=0)
    ks = [np.searchsorted(cum[:, k], A[k], side="right") for k in range(Z.shape[1])]
    if Z.shape[1] == 3 and np.ndim(A[1]) and np.ndim(A[2]):
        return np.minimum(np.minimum(ks[0], ks[1][:, None]), ks[2][None, :])
    return np.min(np.broadcast_arrays(*ks), axis=0)


# ---------------------------------------------------------------- weight from scores (no labels)
def passes(theta, ps, pcs, a1, a2):
    """Cases passed by the cut-off on (1 - theta) ps + theta pcs, for as long as both constraints hold."""
    n = len(ps)
    o = np.argsort((1 - theta) * ps + theta * pcs, kind="stable")
    k_cov = np.searchsorted(np.cumsum(ps[o]) / n, a1, side="right")
    k_sev = np.searchsorted(np.cumsum(pcs[o]) / n, a2, side="right")
    return int(min(k_cov, k_sev))


def best_weight(ps, pcs, a_sev):
    """Weight that passes the most cases, on a coarse grid refined twice. Uses no label.
    By linear-programming duality it equals the weight from the dual variables of the plug-in linear program."""
    a1 = A_COV * ps.mean()
    best_t, best_v = 0.0, -1
    for lo, hi, step in [(0.0, 1.0, 0.02), (None, None, 0.001), (None, None, 0.0001)]:
        if lo is None:
            lo, hi = max(0.0, best_t - 25 * step), min(1.0, best_t + 25 * step)
        for th in np.arange(lo, hi + step / 2, step):
            v = passes(th, ps, pcs, a1, a_sev)
            if v > best_v:
                best_t, best_v = float(th), v
    return best_t


# ---------------------------------------------------------------- calibration (multi-risk CRC)
def crc_cut(score, loss, alpha, bound):
    """sup{t : (sum_i loss_i 1{score_i < t} + bound) / (n + 1) <= alpha}. The rule passes score < t.
    alpha may be an array (one cut-off per level)."""
    n = len(score)
    o = np.argsort(score, kind="stable")
    s = score[o]
    cum = np.concatenate([[0.0], np.cumsum(loss[o])])
    b = np.atleast_1d(alpha) * (n + 1) - bound
    k = np.searchsorted(cum, b + 1e-12, side="right") - 1           # cases passed on the calibration sample
    t = np.where(k < 0, -np.inf, np.where(k >= n, np.inf, s[np.clip(k, 0, n - 1)]))
    return t if np.ndim(alpha) else float(t[0])


def cut_cond(score, adverse, alpha):
    """Conditional constraint Pr(pass | adverse) <= alpha: CRC with loss 1 on the adverse calibration cases only."""
    d = score[adverse]
    return crc_cut(d, np.ones(len(d)), alpha, 1.0)


def crc_cutoff(s_cal, y_cal, loss_cal, a_sev):
    """Two risks. Cut-off t_hat = min(t_cov, t_sev). The rule passes a new case with score s < t_hat."""
    return min(cut_cond(s_cal, y_cal == 1, A_COV), crc_cut(s_cal, loss_cal, a_sev, B))


def cut3(s_cal, y, loss, loss2, a2, a3):
    """Three risks. Cut-off min(t_cov, t_sev, t_late) on one score."""
    return min(cut_cond(s_cal, y == 1, A_COV), crc_cut(s_cal, loss, a2, B), crc_cut(s_cal, loss2, a3, B))


# ---------------------------------------------------------------- shared population quantities
def s_cc_of(p, pc):
    """Severity used by the coverage-only rule (cut-off on p at level alpha_cov)."""
    n = len(p)
    o = np.argsort(p, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]) / n, A_COV * p.mean(), side="right"))
    return np.cumsum(pc[o])[k - 1] / n


def population(big, seed, ratio=RATIO):
    """alpha_sev = ratio x S_sev and the best weight theta*, from a large draw with the true scores."""
    p, c = true_scores(draw_x(big, np.random.default_rng(seed)))
    a_sev = ratio * s_cc_of(p, p * c)
    return a_sev, best_weight(p, p * c, a_sev)


# ---------------------------------------------------------------- part (a)
def part_a(theta_star, a_sev, reps, seed):
    sizes = [1_000, 4_000, 16_000, 64_000, 256_000]
    rng = np.random.default_rng(seed + 1)
    err = np.empty((len(sizes), reps))
    for i, n_u in enumerate(sizes):
        t0 = time.time()
        for r in range(reps):
            p, c = true_scores(draw_x(n_u, rng))
            err[i, r] = abs(best_weight(p, p * c, a_sev) - theta_star)
        print(f"  (a) n_U = {n_u:7d}: mean |theta_hat - theta*| = {err[i].mean():.4f}  ({time.time() - t0:.0f} s)",
              flush=True)
    return np.array(sizes), err


# ---------------------------------------------------------------- part (c)
def part_c(theta_star, a_sev, reps, n_eval, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    sizes = [2_000, 5_000, 20_000, 50_000, 200_000]
    rng = np.random.default_rng(seed + 2)

    # evaluation sample with the true scores. Shares are expectations over X, so the true p is enough.
    x_ev = draw_x(n_eval, rng)
    p_ev, c_ev = true_scores(x_ev)
    pc_ev = p_ev * c_ev
    pi1 = p_ev.mean()
    a1_ev = A_COV * pi1
    best = passes(theta_star, p_ev, pc_ev, a1_ev, a_sev)
    r_star = 1 - best / n_eval                                           # smallest share sent to review

    def evaluate(score_ev, t_hat):
        passed = score_ev < t_hat
        review = 1 - passed.mean()
        cov = (p_ev * passed).mean() / pi1                               # Pr(pass | Y = 1)
        sev = (pc_ev * passed).mean()                                    # E[C 1{Y = 1, pass}]
        return review, cov, sev

    out = {k: np.empty((len(sizes), reps)) for k in ["cal", "wt", "full", "cov", "sev"]}
    for i, n in enumerate(sizes):
        t0 = time.time()
        for r in range(reps):
            # three independent samples of size n: training, unlabeled, calibration
            x_tr, x_un, x_ca = draw_x(n, rng), draw_x(n, rng), draw_x(n, rng)
            p_tr, c_tr = true_scores(x_tr)
            y_tr, l_tr = draw_labels(p_tr, c_tr, rng)
            p_un, c_un = true_scores(x_un)
            p_ca, c_ca = true_scores(x_ca)
            y_ca, l_ca = draw_labels(p_ca, c_ca, rng)

            # estimated scores, fitted on the training sample only
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=r).fit(x_tr, y_tr)
            d = y_tr == 1
            reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=r).fit(x_tr[d], l_tr[d])
            ph = lambda xx: clf.predict_proba(xx)[:, 1]
            ch = lambda xx: np.clip(reg.predict(xx), 0, B)

            # run 1: true scores, theta*
            h_ca = (1 - theta_star) * p_ca + theta_star * p_ca * c_ca
            h_ev = (1 - theta_star) * p_ev + theta_star * pc_ev
            rev1, _, _ = evaluate(h_ev, crc_cutoff(h_ca, y_ca, l_ca, a_sev))

            # run 2: true scores, theta_hat from the unlabeled sample
            th2 = best_weight(p_un, p_un * c_un, a_sev)
            h_ca = (1 - th2) * p_ca + th2 * p_ca * c_ca
            h_ev = (1 - th2) * p_ev + th2 * pc_ev
            rev2, _, _ = evaluate(h_ev, crc_cutoff(h_ca, y_ca, l_ca, a_sev))

            # run 3: the practical pipeline. estimated scores, theta_hat from them, calibrated cut-off
            pu, cu = ph(x_un), ch(x_un)
            th3 = best_weight(pu, pu * cu, a_sev)
            pca, cca = ph(x_ca), ch(x_ca)
            pev, cev = ph(x_ev), ch(x_ev)
            h_ca = (1 - th3) * pca + th3 * pca * cca
            h_ev = (1 - th3) * pev + th3 * pev * cev
            rev3, cov3, sev3 = evaluate(h_ev, crc_cutoff(h_ca, y_ca, l_ca, a_sev))

            out["cal"][i, r] = rev1 - r_star
            out["wt"][i, r] = rev2 - rev1
            out["full"][i, r] = rev3 - r_star
            out["cov"][i, r], out["sev"][i, r] = cov3, sev3
        print(f"  (c) n = {n:7d}: extra review of the pipeline = {100 * out['full'][i].mean():.2f} pp "
              f"({time.time() - t0:.0f} s)", flush=True)
    return np.array(sizes), out, r_star


# ---------------------------------------------------------------- plotting
def plot(store, path):
    """Two panels: (a) sampling error of the weight, (b) extra review of the full method."""
    k = 2
    fig, axes = plt.subplots(1, k, figsize=(6.5, 2.7))   # printed at the full text width
    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK, labelsize=8 if k == 2 else 7)
    letters = "abc"

    if "a_sizes" in store:
        sizes, err = store["a_sizes"], store["a_err"]
        x, y = np.log(sizes), np.log(err.mean(axis=1))
        ax = axes[0]
        ax.plot(x, y[0] - 0.5 * (x - x[0]), color=MUTED, ls=":", lw=1.3)        # slope -1/2, explained in caption
        ax.plot(x, y, color=BLUE, marker="o", ms=4, lw=1.4, label=r"$\log|\hat\theta-\theta^*|$")
        ax.set_xlabel(f"$\\log n_U$\n({letters[0]})", fontsize=9, color=INK, linespacing=1.8)
        ax.set_ylabel(r"$\log|\hat\theta-\theta^*|$", fontsize=9, color=INK)
        ax.legend(fontsize=8 if k == 2 else 7, frameon=False)
    else:
        axes[0].set_visible(False)

    if "c_sizes" in store:
        sizes = store["c_sizes"]
        full = 100 * store["c_full"].mean(axis=1)
        se = 100 * store["c_full"].std(axis=1, ddof=1) / np.sqrt(store["c_full"].shape[1])
        ax = axes[-1]
        ax.plot(np.log(sizes), full, color=INK, marker="o", ms=4, lw=1.4)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(f"$\\log n$\n({letters[k - 1]})", fontsize=9, color=INK, linespacing=1.8)
        ax.set_ylabel("extra share sent to review (pp)", fontsize=9, color=INK)
        print(f"Panel ({letters[k - 1]}): n, extra review (pp), SE (pp)  [SE goes in the caption]")
        for n_, f_, s_ in zip(sizes, full, se):
            print(f"  {int(n_):7d}  {f_:6.2f}  {s_:5.2f}")
    else:
        axes[-1].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=300)


# ---------------------------------------------------------------- Figure 1(a) helpers
def largest_prefix(score, p, pc, a1, a2):
    """Pass the cases with the lowest score for as long as both constraints hold."""
    n = len(score)
    order = np.argsort(score, kind="stable")
    cov = np.cumsum(p[order]) / n
    sev = np.cumsum(pc[order]) / n
    ok = (cov <= a1) & (sev <= a2)
    k = n if ok.all() else int(np.argmin(ok))
    passed = np.zeros(n, bool)
    passed[order[:k]] = True
    return passed


def single_cut(score, use, level):
    """Cut-off on score that meets one constraint, given by the per-case use and its level."""
    n = len(score)
    order = np.argsort(score, kind="stable")
    k = int(np.searchsorted(np.cumsum(use[order]) / n, level, side="right"))
    return score[order[k - 1]]


# ---------------------------------------------------------------- Figure 1(a)
def fig1a(args):
    """Figure 1(a). Pass regions of four rules in the plane of p and pc. Also prints S_sev and a_low."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    p, c = true_scores(draw_x(args.n, rng))
    pc = p * c
    n = args.n
    pi1 = p.mean()
    a_cov = 0.10
    a1 = a_cov * pi1                              # coverage in linear form, E[phi p] <= alpha_cov pi_1

    # S_sev: uncovered severity of the coverage-only rule (cut-off on p)
    t_cc = single_cut(p, p, a1)
    S_sev = pc[p <= t_cc].sum() / n
    a2 = args.ratio * S_sev                        # alpha_sev

    # lower threshold: largest alpha_sev at which the severity-only rule also meets coverage
    order = np.argsort(pc, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[order]) / n, a1, side="right"))
    a_low = np.cumsum(pc[order])[k - 1] / n

    rules = {}
    rules["p"] = largest_prefix(p, p, pc, a1, a2)                       # theta = 0
    rules["pc"] = largest_prefix(pc, p, pc, a1, a2)                     # theta = 1
    t1 = single_cut(p, p, a1)                                           # coverage threshold on p
    t2 = single_cut(pc, pc, a2)                                         # severity threshold on pc
    rules["rect"] = (p <= t1) & (pc <= t2)                              # two cut-offs, calibrated separately
    best = None
    for th in np.round(np.arange(0, 1.0001, 0.01), 2):                 # optimal weight on a grid
        passed = largest_prefix((1 - th) * p + th * pc, p, pc, a1, a2)
        if best is None or passed.sum() > best[1].sum():
            best = (th, passed)
    th_star, rules["opt"] = best

    tot = pc.mean()
    print(f"alpha_cov = {a_cov:.2f}, pi_1 = {pi1:.4f}")
    print(f"S_sev      = {S_sev:.5f}  ({100 * S_sev / tot:.2f}% of default loss)")
    print(f"alpha_sev = {a2:.5f}  ({100 * a2 / tot:.2f}% of default loss, {args.ratio:.2f} S_sev)")
    print(f"a_low     = {a_low:.5f}  ({100 * a_low / tot:.2f}% of default loss, {a_low / S_sev:.3f} S_sev)")
    print(f"theta*    = {th_star:.2f}")
    names = {"p": "Probability only (theta=0)", "pc": "Expected loss only (theta=1)",
             "rect": "Two cut-offs on p and pc", "opt": f"Blend h_theta (theta*={th_star:.2f})"}
    print(f"\n{'rule':34s} {'review %':>9s} {'coverage used':>14s} {'severity used':>14s}")
    for key in ["p", "pc", "rect", "opt"]:
        v = rules[key]
        print(f"{names[key]:34s} {100 * (1 - v.mean()):9.1f} "
              f"{(p * v).sum() / n / pi1:14.3f} {(pc * v).sum() / n / a2:14.2f}")

    # boundaries in the (p, pc) plane
    tp = p[rules["p"]].max()
    tpc = pc[rules["pc"]].max()
    h = (1 - th_star) * p + th_star * pc
    t_opt = h[rules["opt"]].max()
    rev = {key: 100 * (1 - v.mean()) for key, v in rules.items()}

    idx = rng.choice(n, 20_000, replace=False)
    xmax, ymax = 0.2, 0.06
    fig, ax = plt.subplots(figsize=(3.4, 2.75))   # printed at half the text width
    ax.scatter(p[idx], pc[idx], s=0.8, c=CLOUD, lw=0, zorder=1)

    # diagonal pc = p: no case lies above it, since c <= B = 1
    if not args.no_diagonal:
        ax.plot([0, ymax], [0, ymax], color=MUTED, lw=0.8, ls=":", zorder=2)

    # probability only: vertical line p = tp, drawn up to the diagonal (pc <= p = tp)
    ax.plot([tp, tp], [0, min(tp, ymax)], color=COL["p"], lw=2, zorder=3,
            label=rf"$\theta=0$: {rev['p']:.1f}%")

    # expected loss only: horizontal line pc = tpc, drawn from the diagonal (p >= tpc) to the right
    ax.plot([tpc, xmax], [tpc, tpc], color=COL["pc"], lw=2, zorder=3,
            label=rf"$\theta=1$: {rev['pc']:.1f}%")

    # two cut-offs: rectangle p <= t1 and pc <= t2, top edge drawn from the diagonal (p >= t2)
    ax.plot([t1, t1, min(t2, t1)], [0, min(t2, t1), min(t2, t1)], color=COL["rect"], lw=2, zorder=3,
            label=rf"two cut-offs: {rev['rect']:.1f}%")

    # blend: line (1 - theta) p + theta pc = t_opt, from the diagonal (p = t_opt) to the axis (pc = 0)
    if 0 < th_star < 1:
        x_start = t_opt                              # meets the diagonal pc = p
        x_end = min(t_opt / (1 - th_star), xmax)     # meets the axis pc = 0
        xs = np.linspace(x_start, x_end, 200)
        ax.plot(xs, (t_opt - (1 - th_star) * xs) / th_star, color=COL["opt"], lw=3, zorder=4,
                label=rf"$\theta^*={th_star:.2f}$: {rev['opt']:.1f}%")
    elif th_star == 0:
        ax.plot([t_opt, t_opt], [0, min(t_opt, ymax)], color=COL["opt"], lw=3, zorder=4,
                label=rf"Blend $h_\theta$, $\theta^*=0$: review {rev['opt']:.1f}%")
    else:
        ax.plot([t_opt, xmax], [t_opt, t_opt], color=COL["opt"], lw=3, zorder=4,
                label=rf"Blend $h_\theta$, $\theta^*=1$: review {rev['opt']:.1f}%")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("default probability $p(x)$", fontsize=8.5, color=INK)
    ax.set_ylabel("expected loss $p(x)c(x)$", fontsize=8.5, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=7.5)
    ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.95, edgecolor=GRID,
              title="sent to review", title_fontsize=7, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(out / "fig_shape.png", dpi=300)
    print(f"\nSaved {out / 'fig_shape.png'}")


# ---------------------------------------------------------------- Figure 2
def fig2(args):
    """Parts a (sampling error of the weight from cases without outcomes) and c (full method)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_path = out / "fig_theorem2_data.npz"
    fig_path = out / "fig_theorem2.png"

    # results saved by earlier runs, so that parts can be computed separately and redrawn
    store = dict(np.load(data_path, allow_pickle=True)) if data_path.exists() else {}
    if args.plot_only:
        if not store:
            raise SystemExit(f"No saved results at {data_path}. Run without --plot-only first.")
        plot(store, fig_path)
        print(f"Redrew {fig_path} from {data_path}")
        return

    t0 = time.time()
    a_sev, _ = population(args.big, args.seed)
    # theta* with little sampling noise, from a larger independent draw
    p_r, c_r = true_scores(draw_x(args.ref, np.random.default_rng(args.seed + 10)))
    theta_star = best_weight(p_r, p_r * c_r, a_sev)
    del p_r, c_r
    print(f"alpha_cov = {A_COV}, alpha_sev = {a_sev:.5f} ({RATIO} S_sev), theta* = {theta_star:.4f} "
          f"({time.time() - t0:.0f} s)\n", flush=True)

    if "a" in args.parts:
        sizes, err = part_a(theta_star, a_sev, args.reps_a, args.seed)
        store["a_sizes"], store["a_err"] = sizes, err
        mean = err.mean(axis=1)
        se = err.std(axis=1, ddof=1) / np.sqrt(err.shape[1])
        print("\n(a) n_U, mean |theta_hat - theta*|, 2 SE, mean x sqrt(n_U)")
        for n_u, m_, s_ in zip(sizes, mean, se):
            print(f"  {n_u:7d}  {m_:.5f}  {2 * s_:.5f}  {m_ * np.sqrt(n_u):.3f}")
        slope = np.polyfit(np.log(sizes), np.log(mean), 1)[0]
        print(f"  log-log slope: {slope:.2f}  (root-n rate: -0.5)\n")

    if "c" in args.parts:
        sizes, res, r_star = part_c(theta_star, a_sev, args.reps_c, args.eval, args.seed)
        store["c_sizes"] = sizes
        for k in ["cal", "wt", "full", "cov", "sev"]:
            store["c_" + k] = res[k]
        cal = 100 * res["cal"].mean(axis=1)
        wt = 100 * res["wt"].mean(axis=1)
        full = 100 * res["full"].mean(axis=1)
        sc = full - cal - wt
        se_full = 100 * res["full"].std(axis=1, ddof=1) / np.sqrt(res["full"].shape[1])
        print(f"\n(c) optimal share sent to review R* = {100 * r_star:.2f}%")
        print("  n        calibration  weight   score    pipeline (2 SE)   coverage (<= 0.10)  severity (<= alpha_sev)")
        for i, n in enumerate(sizes):
            print(f"  {n:7d}  {cal[i]:10.3f}  {wt[i]:7.3f}  {sc[i]:7.3f}  {full[i]:8.3f} ({2 * se_full[i]:.3f})"
                  f"  {res['cov'][i].mean():12.4f}  {res['sev'][i].mean():14.6f}")
        print(f"  alpha_sev = {a_sev:.6f}. Parts can be negative in a single size because of noise.\n")

    np.savez(data_path, **store)
    plot(store, fig_path)
    print(f"Saved {fig_path} and {data_path}  (total {time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- two risks: positive prices and gain per level
def levels_fig(args):
    """Two risks, true p and c, against alpha_sev (population version, no calibration). For every level: the prices of
    the linear program and how many are positive, and the share passed by four rules. Each rule passes the cases with
    the lowest score for as long as both constraints hold. Rules: best weight theta*, theta = 0, theta = 1, separate
    cut-offs. (a) share of the less-used level that each rule uses, (b) share passed."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    p, c = true_scores(draw_x(args.n, np.random.default_rng(args.seed)))
    pc = p * c
    n = args.n
    a1 = A_COV * p.mean()                                             # coverage in linear form

    o = np.argsort(p, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]) / n, a1, side="right"))
    S_sev = np.cumsum(pc[o])[k - 1] / n
    o = np.argsort(pc, kind="stable")
    k = int(np.searchsorted(np.cumsum(p[o]) / n, a1, side="right"))
    a_low = np.cumsum(pc[o])[k - 1] / n
    levels = np.geomspace(0.5 * a_low, 1.5 * S_sev, args.levels)

    # prices of the linear program per level, and which are positive
    Z = np.column_stack([p, pc])
    lam = np.array([lp_prices(Z, (a1, a))[0] for a in levels])
    pos = lam * Z.mean(axis=0) > TOL
    n_pos = pos.sum(axis=1)
    print(f"linear programs solved ({time.time() - t0:.0f} s)", flush=True)

    thetas = np.round(np.linspace(0, 1, int(round(1 / args.step)) + 1), 6)
    K = np.empty((len(thetas), len(levels)), int)                    # cases passed
    F = np.empty_like(K, dtype=float)                                 # share of the coverage level used
    S = np.empty_like(K, dtype=float)                                 # share of the severity level used
    for i, th in enumerate(thetas):
        o = np.argsort((1 - th) * p + th * pc, kind="stable")
        cp_, cs_ = np.cumsum(p[o]) / n, np.cumsum(pc[o]) / n
        kk = np.minimum(np.searchsorted(cp_, a1, side="right"), np.searchsorted(cs_, levels, side="right"))
        K[i] = kk
        F[i] = np.where(kk > 0, cp_[np.maximum(kk - 1, 0)], 0) / a1
        S[i] = np.where(kk > 0, cs_[np.maximum(kk - 1, 0)], 0) / levels

    # best weight: an end if the best block reaches it (one price is zero), else the centre of the block
    best = np.empty(len(levels), int)
    for j in range(len(levels)):
        b = np.flatnonzero(K[:, j] == K[:, j].max())
        if b[0] == 0:
            best[j] = 0
        elif b[-1] == len(thetas) - 1:
            best[j] = len(thetas) - 1
        else:
            # several grid weights pass the same number of cases. The grid cannot hit theta* exactly, so pick the
            # one closest to using both levels in full (largest share of the less-used level)
            best[j] = b[np.argmax(np.minimum(F[b, j], S[b, j]))]
    cols = np.arange(len(levels))
    theta_opt = thetas[best].astype(float)
    kF, fF, sF = K[best, cols].astype(float), F[best, cols], S[best, cols]

    # where both prices are positive, theta* is the weight at which the two constraints stop the same prefix.
    # A grid cannot hit it exactly, so it is found by bisection, which removes the wiggle of the grid.
    def prefix(th, j):
        o = np.argsort((1 - th) * p + th * pc, kind="stable")
        cp_, cs_ = np.cumsum(p[o]) / n, np.cumsum(pc[o]) / n
        kc = int(np.searchsorted(cp_, a1, side="right"))
        ks = int(np.searchsorted(cs_, levels[j], side="right"))
        k = min(kc, ks)
        return kc, ks, k, (cp_[k - 1] / a1 if k else 0.0), (cs_[k - 1] / levels[j] if k else 0.0)

    for j in range(len(levels)):
        if 0 < best[j] < len(thetas) - 1:
            lo, hi = 0.0, 1.0                    # at lo severity stops first, at hi coverage stops first
            for _ in range(int(args.bisect)):
                mid = (lo + hi) / 2
                kc, ks, *_ = prefix(mid, j)
                if ks < kc:
                    lo = mid
                else:
                    hi = mid
            _, _, k, f, s_ = prefix((lo + hi) / 2, j)
            theta_opt[j], kF[j], fF[j], sF[j] = (lo + hi) / 2, k, f, s_
    res = {"opt": (kF, fF, sF),
           "p": (K[0], F[0], S[0]), "pc": (K[-1], F[-1], S[-1])}
    t1 = single_cut(p, p, a1)                                         # separate cut-offs, one per constraint
    rect = [(p <= t1) & (pc <= single_cut(pc, pc, a)) for a in levels]
    res["rect"] = (np.array([r.sum() for r in rect]), np.array([(p * r).sum() / n / a1 for r in rect]),
                   np.array([(pc * r).sum() / n / a for r, a in zip(rect, levels)]))
    gain_single = 100 * (kF - np.maximum(K[0], K[-1])) / n            # pp, over the better single score
    gain_rect = 100 * (kF - res["rect"][0]) / n                       # pp, over separate cut-offs

    names = {"opt": r"best weight $\theta^*$", "p": r"probability only, $\theta=0$",
             "pc": r"expected loss only, $\theta=1$", "rect": "separate cut-offs"}
    rows = {"alpha_sev": levels, "lambda_cov": lam[:, 0], "lambda_sev": lam[:, 1], "n_pos": n_pos,
            "theta_star": theta_opt, "gain_vs_single": gain_single, "gain_vs_separate": gain_rect}
    for key, (kk, f, s) in res.items():
        rows[f"accept_{key}"], rows[f"freq_used_{key}"], rows[f"loss_used_{key}"] = 100 * kk / n, 100 * f, 100 * s
    with open(out / "fig_levels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(rows.keys())
        w.writerows(zip(*rows.values()))

    print(f"a_low = {a_low:.5f}, S_sev = {S_sev:.5f}, {len(levels)} levels from {levels[0]:.5f} to {levels[-1]:.5f}")
    print(f"\n{'alpha_sev':>10s} {'+prices':>7s} {'gain':>6s} " + " ".join(f"{k + ' acc/freq/loss':>24s}" for k in res))
    for j in np.linspace(0, len(levels) - 1, 12).astype(int):
        tag = "".join("cs"[k] if pos[j, k] else "-" for k in range(2))
        print(f"{levels[j]:10.5f} {tag:>7s} {gain_single[j]:6.2f} " + " ".join(
            f"{100 * kk[j] / n:8.1f} {100 * f[j]:6.0f} {100 * s[j]:6.0f}  " for kk, f, s in res.values()))
    two = n_pos == 2
    print(f"\nlevels with two positive prices: {int(two.sum())} of {len(levels)}"
          + (f", alpha_sev from {levels[two].min():.5f} to {levels[two].max():.5f}" if two.any() else ""))
    print(f"gain of theta* over the better single score (pp): with two positive prices mean "
          f"{gain_single[two].mean() if two.any() else np.nan:.2f}, max {gain_single[two].max() if two.any() else np.nan:.2f}; "
          f"with one, largest |gain| {np.abs(gain_single[~two]).max() if (~two).any() else np.nan:.3f}")
    print(f"gain over separate cut-offs (pp): with two positive prices mean "
          f"{gain_rect[two].mean() if two.any() else np.nan:.2f}; with one, mean "
          f"{gain_rect[~two].mean() if (~two).any() else np.nan:.2f}")

    # uniqueness check: weights within one case of the most passed form one block
    widths, blocks = [], []
    for j in range(len(levels)):
        near = np.flatnonzero(K[:, j] >= K[:, j].max() - 1)
        blocks.append(1 + int(np.sum(np.diff(near) > 1)))
        widths.append(thetas[near[-1]] - thetas[near[0]])
    print(f"uniqueness: levels with a single block of best weights {sum(b == 1 for b in blocks)} of {len(levels)}, "
          f"largest width {max(widths):.4f} (grid step {args.step:g})")

    col = {"opt": COL["opt"], "p": PURPLE, "pc": COL["pc"], "rect": COL["rect"]}
    LW = 1.4
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0))               # printed at the full text width
    for key in ["p", "pc", "rect", "opt"]:                           # theta* last, so that it lies on top
        kk, f, s = res[key]
        z = 3 if key == "opt" else 2
        # (a) used share of the less-used level, min(share of coverage level used, share of severity level used)
        axes[0].plot(levels, np.clip(100 * np.minimum(f, s), 0, 100), color=col[key], lw=LW, zorder=z)
        # (b) share passed
        axes[1].plot(levels, 100 * np.asarray(kk) / n, color=col[key], lw=LW, zorder=z)
    axes[0].set_ylabel("less-used level, share used (%)", fontsize=8.5, color=INK)
    axes[0].set_ylim(-3, 103)
    axes[1].set_ylabel("share passed (%)", fontsize=8.5, color=INK)
    for ax, letter in zip(axes, "ab"):
        if two.any():                                                 # levels with two positive prices
            ax.axvspan(levels[two].min(), levels[two].max(), color=BAND, lw=0, zorder=0)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(r"severity level $\alpha_{\mathrm{sev}}$" + f"\n({letter})", fontsize=8.5, color=INK,
                      linespacing=1.7)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=INK, labelsize=7)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=col[k], lw=LW, label=names[k]) for k in ["opt", "p", "pc", "rect"]]
    handles += [Patch(color=BAND, label="two positive prices")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.0),
               handlelength=1.8, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=2.0)
    fig.savefig(out / "fig_levels.png", dpi=300)
    print(f"\nSaved {out / 'fig_levels.png'} and {out / 'fig_levels.csv'}  ({time.time() - t0:.0f} s)")


# ---------------------------------------------------------------- three risks
PAIRS = [(1, 2), (0, 2), (0, 1)]                                   # PAIRS[k]: the two risks left when risk k is dropped
RISK = ["coverage", "severity", "late payment"]


def three_population(n, seed):
    """True Z = (p, pc, q), the coverage level a_1 = alpha_cov pi_1 in linear form, and S_sev, S_late: the severity
    and the late payments used by the coverage-only rule. The grid of levels is set relative to S_sev and S_late."""
    Z = risks3(draw_x(n, np.random.default_rng(seed), 4))
    a1 = A_COV * Z[:, 0].mean()
    o = np.argsort(Z[:, 0], kind="stable")
    k = int(np.searchsorted(np.cumsum(Z[o, 0]) / n, a1, side="right"))
    S = Z[o][:k].sum(axis=0) / n
    return Z, a1, S[1], S[2]


def pair_score(Z, u, v, w, mu):
    return (1 - w) * Z[:, u] / mu[u] + w * Z[:, v] / mu[v]


def best_pair(Z, A, coarse=0.01, fine=0.0005):
    """Best score that uses two of the three risks: over the three pairs and the weight w of
    (1 - w) Z_u / E[Z_u] + w Z_v / E[Z_v], the rule that passes the most cases of Z while all three sums stay
    within A. Uses no labels. Returns (dropped risk, w, cases passed, means used for the scaling)."""
    mu = Z.mean(axis=0)
    best = (None, None, -1)
    for k, (u, v) in enumerate(PAIRS):
        ws = np.arange(0, 1 + coarse / 2, coarse)
        cnt = [prefix_count(pair_score(Z, u, v, w, mu), Z, A) for w in ws]
        w0 = ws[int(np.argmax(cnt))]
        ws = np.arange(max(0.0, w0 - coarse), min(1.0, w0 + coarse) + fine / 2, fine)
        cnt = [prefix_count(pair_score(Z, u, v, w, mu), Z, A) for w in ws]
        if max(cnt) > best[2]:
            best = (k, float(ws[int(np.argmax(cnt))]), int(max(cnt)))
    return best[0], best[1], best[2], mu


def three_map(args, out):
    """Population version on a grid of (alpha_sev, alpha_late), true scores, no calibration."""
    t0 = time.time()
    Z, a1, S2, S3 = three_population(args.n, args.seed)
    n = len(Z)
    r = np.geomspace(args.lo, args.hi, args.grid)                   # levels as multiples of S_sev and S_late
    a2s, a3s = r * S2, r * S3
    m = len(r)
    print(f"Three risks, {n:,} cases. a_1 = {a1:.5f}, S_sev = {S2:.5f}, S_late = {S3:.5f}. "
          f"Grid {m} x {m}, ratios {r[0]:.2f} to {r[-1]:.2f}", flush=True)

    # combined score: prices and share passed from the linear program in every cell
    lam = np.empty((m, m, 3))
    full = np.empty((m, m))
    for i, a2 in enumerate(a2s):
        for j, a3 in enumerate(a3s):
            lam[i, j], full[i, j] = lp_prices(Z, (a1, a2, a3))
        print(f"  linear programs, row {i + 1}/{m}  [{time.time() - t0:.0f}s]", flush=True)
    pos = lam * Z.mean(axis=0) > TOL
    n_pos = pos.sum(axis=2)

    A = (n * a1, n * a2s, n * a3s)
    # scores that drop one risk: best weight on a fine grid, one sort per weight serves all cells
    mu = Z.mean(axis=0)
    drop = np.zeros((3, m, m))
    for k, (u, v) in enumerate(PAIRS):
        for w in np.linspace(0, 1, args.steps + 1):
            drop[k] = np.maximum(drop[k], prefix_count(pair_score(Z, u, v, w, mu), Z, A))
        print(f"  scores without {RISK[k]} done  [{time.time() - t0:.0f}s]", flush=True)
    single = np.stack([prefix_count(Z[:, k], Z, A) for k in range(3)])
    # separate cut-offs: Z_k <= t_k, each t_k set so that constraint k alone holds
    m1 = Z[:, 0] <= single_cut(Z[:, 0], Z[:, 0], a1)
    m2 = [Z[:, 1] <= single_cut(Z[:, 1], Z[:, 1], a) for a in a2s]
    m3 = [Z[:, 2] <= single_cut(Z[:, 2], Z[:, 2], a) for a in a3s]
    sep = np.array([[(m1 & m2[i] & m3[j]).sum() for j in range(m)] for i in range(m)])

    full, drop, single, sep = 100 * full, 100 * drop / n, 100 * single / n, 100 * sep / n   # shares passed (%)
    best_drop, best_single = drop.max(axis=0), single.max(axis=0)
    rows = []
    for i in range(m):
        for j in range(m):
            rows.append(dict(r_sev=r[i], r_late=r[j], alpha_sev=a2s[i], alpha_late=a3s[j],
                             lambda_cov=lam[i, j, 0], lambda_sev=lam[i, j, 1], lambda_late=lam[i, j, 2],
                             pos=",".join(RISK[k] for k in range(3) if pos[i, j, k]), n_pos=n_pos[i, j],
                             mean_cov=mu[0], mean_sev=mu[1], mean_late=mu[2], share_full=full[i, j], share_drop_cov=drop[0, i, j], share_drop_sev=drop[1, i, j],
                             share_drop_late=drop[2, i, j], share_single_cov=single[0, i, j],
                             share_single_sev=single[1, i, j], share_single_late=single[2, i, j],
                             share_separate=sep[i, j], gain_vs_drop=full[i, j] - best_drop[i, j],
                             gain_vs_single=full[i, j] - best_single[i, j], gain_vs_separate=full[i, j] - sep[i, j]))
    with open(out / "three_map.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    g = {"drop": full - best_drop, "single": full - best_single, "sep": full - sep}
    print("\nShare passed by the combined score minus the best alternative (pp), by number of positive prices:")
    print(f"{'positive':>8s} {'cells':>5s} {'drop one risk: mean':>20s} {'max':>6s} {'single: mean':>13s} "
          f"{'separate: mean':>15s}")
    for kpos in range(4):
        sel = n_pos == kpos
        if sel.any():
            print(f"{kpos:8d} {int(sel.sum()):5d} {g['drop'][sel].mean():20.3f} {g['drop'][sel].max():6.3f} "
                  f"{g['single'][sel].mean():13.3f} {g['sep'][sel].mean():15.3f}")
    print("(where some price is zero, the score without that risk is optimal, so the first gain should be about 0)")
    print("\nCells with the largest gain over dropping one risk, per number of positive prices:")
    for kpos in (3, 2, 1):
        sel = np.argwhere(n_pos == kpos)
        if len(sel):
            i, j = sel[np.argmax([g["drop"][a, b] for a, b in sel])]
            print(f"  {kpos}: alpha_sev = {r[i]:.2f} S_sev, alpha_late = {r[j]:.2f} S_late, positive: "
                  f"{rows[i * m + j]['pos']}, full {full[i, j]:.2f}%, best drop-one {best_drop[i, j]:.2f}% "
                  f"(without {RISK[int(np.argmax(drop[:, i, j]))]}), separate {sep[i, j]:.2f}%")

    draw_three_map(out)
    print(f"\nSaved three_map.csv and fig_three_map.png to {out}  [{time.time() - t0:.0f}s]")


def draw_three_map(out):
    """fig_three_map from three_map.csv: gain over the best drop-one-risk score, with the borders between the regions
    of one, two and three positive prices drawn along the cell edges (solid: border of the three-price region)."""
    import pandas as pd
    d = pd.read_csv(out / "three_map.csv")
    r = np.sort(d.r_sev.unique())
    m = len(r)
    gain = d.pivot(index="r_sev", columns="r_late", values="gain_vs_drop").to_numpy()
    n_pos = d.pivot(index="r_sev", columns="r_late", values="n_pos").to_numpy()
    edges = np.geomspace(r[0] / np.sqrt(r[1] / r[0]), r[-1] * np.sqrt(r[1] / r[0]), m + 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.3))
    im = ax.pcolormesh(edges, edges, np.clip(gain, 0, None), cmap="Blues", vmin=0)
    for i in range(m):                            # borders between cells with different numbers of positive prices
        for j in range(m):
            for di, dj in ((0, 1), (1, 0)):
                i2, j2 = i + di, j + dj
                if i2 < m and j2 < m and n_pos[i, j] != n_pos[i2, j2]:
                    solid = 3 in (n_pos[i, j], n_pos[i2, j2])
                    xs = [edges[j + 1]] * 2 if dj else [edges[j], edges[j + 1]]
                    ys = [edges[i], edges[i + 1]] if dj else [edges[i + 1]] * 2
                    ax.plot(xs, ys, color=INK, lw=1.1 if solid else 0.7, ls="-" if solid else "--")
    for kpos in (1, 2, 3):                        # label each region at its cell farthest from the other regions
        cells_k = np.argwhere(n_pos == kpos)
        other = np.argwhere(n_pos != kpos)
        if len(cells_k) and len(other):
            dist = np.min(np.abs(cells_k[:, None, :] - other[None, :, :]).sum(axis=2), axis=1)
            i, j = cells_k[np.argmax(dist)]
            x, ha = (edges[-1] / 1.04, "right") if j == m - 1 else ((edges[0] * 1.04, "left") if j == 0 else (r[j], "center"))
            ax.text(x, r[i], f"{kpos} positive", ha=ha, va="center", fontsize=7,
                    color="white" if gain[i, j] > 0.6 * np.nanmax(gain) else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.ax.tick_params(labelsize=7, colors=INK)
    cb.set_label("gain over the best drop-one-risk score (pp)", fontsize=7.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(edges[0], edges[-1])
    ticks = [t for t in (0.2, 0.5, 1.0) if r[0] <= t <= r[-1]]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"$\alpha_{\mathrm{late}} / S_{\mathrm{late}}$", fontsize=8.5, color=INK)
    ax.set_ylabel(r"$\alpha_{\mathrm{sev}} / S_{\mathrm{sev}}$", fontsize=8.5, color=INK)
    ax.tick_params(colors=INK, labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_three_map.png", dpi=300)
    # cells whose count rests on a small price: the smallest positive share of the score
    if "mean_cov" not in d:
        print("three_map.csv has no mean_* columns; run the map part again for the check of small prices")
        return
    sh = d[["lambda_cov", "lambda_sev", "lambda_late"]].to_numpy()
    print("Cells with a positive price whose share of the score is below 1% (count may rest on a small price):")
    small = []
    for (_, row), lam in zip(d.iterrows(), sh):
        u = lam * np.array([row.get("mean_cov", np.nan), row.get("mean_sev", np.nan), row.get("mean_late", np.nan)])
        if np.all(np.isfinite(u)) and u.sum() > 0:
            w = u / u.sum()
            if ((w > 0) & (w < 0.01)).any():
                small.append(f"  alpha_sev = {row.r_sev:.2f} S_sev, alpha_late = {row.r_late:.2f} S_late, "
                             f"{int(row.n_pos)} positive, shares {np.round(w, 4)}")
    print("\n".join(small) if small else "  none")


def fit_three(x, y, loss, y2, loss2, seed):
    """Fitted p, c and q from one training sample. Returns a function x -> estimated Z = (p, pc, q)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    kw = dict(max_iter=200, learning_rate=0.1, random_state=seed)
    fp = HistGradientBoostingClassifier(**kw).fit(x, y)
    fc = HistGradientBoostingRegressor(**kw).fit(x[y == 1], loss[y == 1])
    fq = HistGradientBoostingClassifier(**kw).fit(x, y2)

    def zhat(xx):
        p, q = fp.predict_proba(xx)[:, 1], fq.predict_proba(xx)[:, 1]
        c = np.clip(fc.predict(xx), 0, B)
        return np.column_stack([p, p * c, q])
    return zhat


THREE_RULES = ["oracle", "plugin", "drop_one", "separate"]
THREE_LABELS = {"oracle": "true scores and prices", "plugin": "plug-in prices", "drop_one": "best drop-one-risk score",
                "separate": "separate cut-offs"}
THREE_COLORS = {"oracle": INK, "plugin": BLUE, "drop_one": ORANGE, "separate": GREEN}


def three_cal(args, out):
    """Fitted scores, prices from cases without outcomes, multi-risk CRC. Per repetition: training, U (features
    only) and calibration samples of size n. Risks are evaluated on a large sample with the true p, pc and q, so each
    repetition gives the risk of the rule given its samples. Rules:
      oracle    true Z and the true prices lambda*, CRC cut-off
      plugin    fitted Z, prices of the linear program on U, CRC cut-off
      drop_one  fitted Z, the best score that uses two of the three risks (chosen on U), CRC cut-off
      separate  fitted Z, one CRC cut-off per risk on its own score, pass only below all three"""
    t0 = time.time()
    cells = [tuple(float(v) for v in cstr.split(",")) for cstr in args.cells]
    Zr, a1r, S2, S3 = three_population(args.ref, args.seed + 10)
    levels = [(r2 * S2, r3 * S3) for r2, r3 in cells]
    lam_star, opt = zip(*[lp_prices(Zr, (a1r, a2, a3)) for a2, a3 in levels])
    pos_star = [lam * Zr.mean(axis=0) > TOL for lam in lam_star]
    w_star = [score_shares(lam, Zr) for lam in lam_star]
    del Zr
    x_ev = draw_x(args.eval, np.random.default_rng(args.seed + 11), 4)
    Z_ev = risks3(x_ev)
    pi1 = Z_ev[:, 0].mean()
    for (r2, r3), lv, ps, ws, o in zip(cells, levels, pos_star, w_star, opt):
        print(f"cell alpha_sev = {r2:g} S_sev = {lv[0]:.5f}, alpha_late = {r3:g} S_late = {lv[1]:.5f}: positive prices "
              f"{','.join(RISK[k] for k in range(3) if ps[k]) or 'none'}, shares of the score {np.round(ws, 3)}, "
              f"optimal share passed {100 * o:.2f}%")
    print(f"[{time.time() - t0:.0f}s]", flush=True)

    rng = np.random.default_rng(args.seed + 20)
    recs = []
    for rep in range(args.reps):
        x_tr, x_u, x_ca = (draw_x(args.n_sample, rng, 4) for _ in range(3))
        zhat = fit_three(x_tr, *labels3(x_tr, rng), seed=rep)
        y, loss, y2, loss2 = labels3(x_ca, rng)
        Zu, Zca, Zev = zhat(x_u), zhat(x_ca), zhat(x_ev)
        Zca_true = risks3(x_ca)
        for ci, (a2, a3) in enumerate(levels):
            def evaluate(rule, passed, **extra):
                recs.append(dict(rep=rep, cell=ci, rule=rule, share=100 * passed.mean(),
                                 used_cov=100 * (Z_ev[:, 0] * passed).mean() / pi1 / A_COV,
                                 used_sev=100 * (Z_ev[:, 1] * passed).mean() / a2,
                                 used_late=100 * (Z_ev[:, 2] * passed).mean() / a3, **extra))

            lam = lam_star[ci]
            evaluate("oracle", Z_ev @ lam < cut3(Zca_true @ lam, y, loss, loss2, a2, a3))
            Au = (A_COV * Zu[:, 0].mean(), a2, a3)
            lam_h, _ = lp_prices(Zu, Au)
            ws = score_shares(lam_h, Zu)
            evaluate("plugin", Zev @ lam_h < cut3(Zca @ lam_h, y, loss, loss2, a2, a3),
                     n_pos=int((lam_h * Zu.mean(axis=0) > TOL).sum()), w_cov=ws[0], w_sev=ws[1], w_late=ws[2])
            k, w, _, mu = best_pair(Zu, len(Zu) * np.array(Au))
            u, v = PAIRS[k]
            evaluate("drop_one", pair_score(Zev, u, v, w, mu) < cut3(pair_score(Zca, u, v, w, mu), y, loss, loss2, a2, a3),
                     dropped=RISK[k])
            t = (cut_cond(Zca[:, 0], y == 1, A_COV), crc_cut(Zca[:, 1], loss, a2, B), crc_cut(Zca[:, 2], loss2, a3, B))
            evaluate("separate", (Zev[:, 0] < t[0]) & (Zev[:, 1] < t[1]) & (Zev[:, 2] < t[2]))
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  {rep + 1}/{args.reps} repetitions  [{time.time() - t0:.0f}s]", flush=True)

    import pandas as pd
    runs = pd.DataFrame(recs)
    runs.to_csv(out / "three_cal_runs.csv", index=False)
    used = ["used_cov", "used_sev", "used_late"]
    g = runs.groupby(["cell", "rule"])
    summ = g[["share"] + used].mean().join((g[["share"] + used].std() / np.sqrt(args.reps)).add_prefix("se_"))
    for u_ in used:
        summ["exceed_" + u_[5:]] = g[u_].apply(lambda v: 100 * (v > 100).mean())
    summ = summ.reset_index()
    summ.to_csv(out / "three_cal_summary.csv", index=False)

    pd.set_option("display.width", 220)
    for ci, ((r2, r3), o) in enumerate(zip(cells, opt)):
        s = summ[summ.cell == ci].set_index("rule").loc[THREE_RULES]
        print(f"\nCell {ci}: alpha_sev = {r2:g} S_sev, alpha_late = {r3:g} S_late, positive prices "
              f"{','.join(RISK[k] for k in range(3) if pos_star[ci][k]) or 'none'}; optimal share passed {100 * o:.2f}%")
        print("  share passed (%), mean risk / level x 100, and % of repetitions above the level:")
        print(s[["share", "se_share"] + used + ["exceed_cov", "exceed_sev", "exceed_late"]].round(2).to_string())
        pl = runs[(runs.cell == ci) & (runs.rule == "plugin")]
        print(f"  plug-in: same positive prices as the truth in {100 * (pl.n_pos == pos_star[ci].sum()).mean():.0f}% "
              f"of repetitions; shares of the score, true {np.round(w_star[ci], 3)}, plug-in mean "
              f"{np.round(pl[['w_cov', 'w_sev', 'w_late']].mean().to_numpy(), 3)} "
              f"(SD {np.round(pl[['w_cov', 'w_sev', 'w_late']].std().to_numpy(), 3)})")
        dr = runs[(runs.cell == ci) & (runs.rule == "drop_one")].dropped.value_counts()
        print("  drop-one score chosen on U leaves out: " + ", ".join(f"{k_} {v_}" for k_, v_ in dr.items()))
    print("(CRC bounds the mean risk over repetitions; single repetitions above the level are expected)")

    # per-repetition distributions at the first cell
    d = runs[runs.cell == 0]
    fig, axes = plt.subplots(1, 4, figsize=(6.5, 2.6))
    for ax, col, lab in zip(axes, ["share"] + used, ["share passed (%)", "coverage used (%)", "severity used (%)",
                                                     "late payment used (%)"]):
        data = [d[d.rule == r_][col].to_numpy() for r_ in THREE_RULES]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for patch, r_ in zip(bp["boxes"], THREE_RULES):
            patch.set_facecolor(THREE_COLORS[r_])
            patch.set_alpha(0.35)
        for med in bp["medians"]:
            med.set_color(INK)
        ax.axhline(100 if col != "share" else 100 * opt[0], color=MUTED, lw=0.8, ls="--")
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["true", "plug-in", "drop one", "separate"], rotation=45, ha="right", fontsize=6.5)
        ax.set_title(lab, fontsize=7.5, color=INK)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=INK, labelsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_three_splits.png", dpi=300)
    print(f"\nSaved three_cal_runs.csv, three_cal_summary.csv and fig_three_splits.png to {out}  "
          f"[{time.time() - t0:.0f}s]")


def simplex_grid(step):
    """Weights (w_1, w_2, w_late) >= 0 with sum 1 on a grid of the given step."""
    m = int(round(1 / step))
    return np.array([(i / m, j / m, (m - i - j) / m) for i in range(m + 1) for j in range(m + 1 - i)])


def labeled_weights(Zn, y, loss, loss2, a2, a3, grid):
    """Weight on the grid whose score Zn @ w passes the most of these labeled cases after CRC on them."""
    best, best_v = grid[0], -1.0
    for w in grid:
        s = Zn @ w
        v = (s < cut3(s, y, loss, loss2, a2, a3)).mean()
        if v > best_v:
            best, best_v = w, v
    return best


def three_labels(args, out):
    """Label budget. m labeled cases, fitted scores. Both rules use the same m labels:
      plugin   prices of the linear program on U (features only); all m labeled cases calibrate the cut-off
      labeled  weights on a grid of the simplex (step --wstep) chosen on m/2 labeled cases (the weight that passes
               the most of them after CRC on them); the other m/2 calibrate the cut-off, so that the guarantees hold
    Scores are scaled by their means on U, so that the grid covers comparable weights."""
    import pandas as pd
    t0 = time.time()
    cells = [tuple(float(v) for v in cstr.split(",")) for cstr in args.label_cells]
    Zr, _, S2, S3 = three_population(args.ref, args.seed + 10)
    levels = [(r2 * S2, r3 * S3) for r2, r3 in cells]
    del Zr
    x_ev = draw_x(args.eval, np.random.default_rng(args.seed + 11), 4)
    Z_ev = risks3(x_ev)
    pi1 = Z_ev[:, 0].mean()
    grid = simplex_grid(args.wstep)
    budgets = sorted(args.budgets)
    print(f"Label budgets {budgets}, {len(grid)} grid weights, cells " +
          ", ".join(f"({r2:g} S_sev, {r3:g} S_late)" for r2, r3 in cells), flush=True)

    rng = np.random.default_rng(args.seed + 40)
    recs = []
    for rep in range(args.reps):
        x_tr, x_u, x_l = draw_x(args.n_sample, rng, 4), draw_x(args.n_sample, rng, 4), draw_x(budgets[-1], rng, 4)
        zhat = fit_three(x_tr, *labels3(x_tr, rng), seed=rep)
        y, loss, y2, loss2 = labels3(x_l, rng)                       # labeled cases in random order
        Zu, Zl, Zev = zhat(x_u), zhat(x_l), zhat(x_ev)
        mu = Zu.mean(axis=0)
        for ci, (a2, a3) in enumerate(levels):
            lam_h, _ = lp_prices(Zu, (A_COV * Zu[:, 0].mean(), a2, a3))
            for m in budgets:
                L = np.arange(m)
                A, Bh = L[: m // 2], L[m // 2:]

                def add(rule, passed, cal):
                    recs.append(dict(rep=rep, cell=ci, m=m, rule=rule, share=100 * passed.mean(),
                                     used_cov=100 * (Z_ev[:, 0] * passed).mean() / pi1 / A_COV,
                                     used_sev=100 * (Z_ev[:, 1] * passed).mean() / a2,
                                     used_late=100 * (Z_ev[:, 2] * passed).mean() / a3,
                                     cal_adverse=int(y[cal].sum()), cal_late=int(y2[cal].sum())))

                add("plugin", Zev @ lam_h < cut3(Zl[L] @ lam_h, y[L], loss[L], loss2[L], a2, a3), L)
                w = labeled_weights(Zl[A] / mu, y[A], loss[A], loss2[A], a2, a3, grid)
                add("labeled", (Zev / mu) @ w < cut3((Zl[Bh] / mu) @ w, y[Bh], loss[Bh], loss2[Bh], a2, a3), Bh)
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  {rep + 1}/{args.reps} repetitions  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    runs.to_csv(out / "three_labels_runs.csv", index=False)
    used = ["used_cov", "used_sev", "used_late"]
    g = runs.groupby(["cell", "m", "rule"])
    summ = g[["share"] + used + ["cal_adverse", "cal_late"]].mean().join(
        (g[["share"] + used].std() / np.sqrt(args.reps)).add_prefix("se_"))
    summ["none_passed"] = g.share.apply(lambda v: 100 * (v == 0).mean())
    summ = summ.reset_index()
    w_ = runs.pivot_table(index=["rep", "cell", "m"], columns="rule", values="share")
    d = (w_["plugin"] - w_["labeled"]).groupby(level=["cell", "m"])
    diff = pd.DataFrame({"diff": d.mean(), "se_diff": d.std() / np.sqrt(args.reps)}).reset_index()
    summ = summ.merge(diff, on=["cell", "m"])
    summ.to_csv(out / "three_labels_summary.csv", index=False)

    pd.set_option("display.width", 220)
    for ci, (r2, r3) in enumerate(cells):
        s = summ[summ.cell == ci]
        print(f"\nCell {ci}: alpha_sev = {r2:g} S_sev, alpha_late = {r3:g} S_late. Share passed (%), plug-in minus labeled "
              f"(pp, paired), mean risk / level x 100, adverse cases in the calibration sample, % passing no one:")
        print(s.pivot(index="m", columns="rule", values=["share", "used_cov", "used_sev", "used_late", "cal_adverse",
                                                         "cal_late", "none_passed"]).round(1).to_string())
        print(s[s.rule == "plugin"].set_index("m")[["diff", "se_diff"]].round(2).T.to_string())
    print("(CRC bounds the mean risk over repetitions)")

    fig, axes = plt.subplots(1, len(cells), figsize=(6.5, 2.7), sharey=True, squeeze=False)
    for ax, (ci, (r2, r3)) in zip(axes[0], enumerate(cells)):
        s = summ[summ.cell == ci]
        for rule, col, lab in [("plugin", BLUE, "plug-in prices, no labels"),
                               ("labeled", PURPLE, "weights chosen on half the labels")]:
            v = s[s.rule == rule].sort_values("m")
            ax.plot(v.m, v.share, color=col, marker="o", ms=3, lw=1.3, label=lab)
        ax.set_xscale("log")
        ax.set_xlabel(f"labeled cases m\n({'ab'[ci]}) " + r"$\alpha_{\mathrm{sev}}$" + f" = {r2:g} $S_{{\\mathrm{{sev}}}}$, "
                      + r"$\alpha_{\mathrm{late}}$" + f" = {r3:g} $S_{{\\mathrm{{late}}}}$", fontsize=8, color=INK, linespacing=1.7)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=INK, labelsize=7)
    axes[0, 0].set_ylabel("share passed (%)", fontsize=8.5, color=INK)
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_three_labels.png", dpi=300)
    print(f"\nSaved three_labels_runs.csv, three_labels_summary.csv and fig_three_labels.png to {out}  "
          f"[{time.time() - t0:.0f}s]")


def three(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if "map" in args.parts:
        three_map(args, out)
    if "mapfig" in args.parts:
        draw_three_map(out)
    if "cal" in args.parts:
        three_cal(args, out)
    if "labels" in args.parts:
        three_labels(args, out)


# ---------------------------------------------------------------- value of better estimates (two risks)
ERRORS = {"p bias": ("p", "bias"), "c bias": ("c", "bias"), "p noise": ("p", "noise"), "c noise": ("c", "noise")}
ERR_COL = {"p": PURPLE, "c": BLUE}


def distort(p, c, kind, size, rng):
    """Estimates with an error of the given size on the logit scale. bias: logit + size, the same for every case
    (size < 0 under-estimates, size > 0 over-estimates). noise: logit + size N(0, 1), drawn per case."""
    which, how = ERRORS[kind]
    z = logit(p if which == "p" else c)
    z = z + size * rng.standard_normal(len(z)) if how == "noise" else z + size
    return (sigmoid(z), c) if which == "p" else (p, sigmoid(z))


def errors(args):
    """Two risks only (coverage and severity), errors in p and c. The weight comes from U (features only) and the
    cut-off from calibration cases with outcomes, as in the method, but both use estimates with a controlled error. Share passed and risks are evaluated with the true
    p and pc on a large sample. The difference to size 0 is what removing the error is worth."""
    import pandas as pd
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    x_ev = draw_x(args.eval, np.random.default_rng(args.seed + 11))
    p_ev, c_ev = true_scores(x_ev)
    pc_ev, pi1 = p_ev * c_ev, p_ev.mean()
    targets = []
    for ratio in args.ratios:
        a_sev, th_star = population(args.big, args.seed, ratio)
        best = passes(th_star, p_ev, pc_ev, A_COV * pi1, a_sev) / args.eval
        targets.append((ratio, a_sev, th_star, best))
        print(f"alpha_sev = {ratio:g} S_sev = {a_sev:.5f}: theta* = {th_star:.4f}, optimal share passed "
              f"{100 * best:.2f}%", flush=True)

    rng = np.random.default_rng(args.seed + 30)
    recs = []
    configs = [("none", 0.0)] + [(k, s) for k in ERRORS for s in (args.biases if "bias" in k else args.noises)
                                 if s != 0]
    for rep in range(args.reps):
        x_u, x_ca = draw_x(args.n_u, rng), draw_x(args.n_cal, rng)
        p_u, c_u = true_scores(x_u)
        p_ca, c_ca = true_scores(x_ca)
        y_ca, l_ca = draw_labels(p_ca, c_ca, rng)
        for kind, size in configs:
            if kind == "none":
                (pu, cu), (pca, cca), (pev, cev) = (p_u, c_u), (p_ca, c_ca), (p_ev, c_ev)
            else:
                pu, cu = distort(p_u, c_u, kind, size, rng)
                pca, cca = distort(p_ca, c_ca, kind, size, rng)
                pev, cev = distort(p_ev, c_ev, kind, size, rng)
            mae_p, mae_pc = np.abs(pev - p_ev).mean(), np.abs(pev * cev - pc_ev).mean()
            for ratio, a_sev, _, best in targets:
                th = best_weight(pu, pu * cu, a_sev)
                s_ca = (1 - th) * pca + th * pca * cca
                passed = (1 - th) * pev + th * pev * cev < crc_cutoff(s_ca, y_ca, l_ca, a_sev)
                recs.append(dict(rep=rep, ratio=ratio, kind=kind, size=size, theta=th, mae_p=mae_p, mae_pc=mae_pc,
                                 share=100 * passed.mean(), optimal=100 * best,
                                 used_cov=100 * (p_ev * passed).mean() / pi1 / A_COV,
                                 used_sev=100 * (pc_ev * passed).mean() / a_sev))
        if (rep + 1) % 10 == 0 or rep == 0:
            print(f"  {rep + 1}/{args.reps} repetitions  [{time.time() - t0:.0f}s]", flush=True)

    runs = pd.DataFrame(recs)
    # size 0 belongs to every error type
    base = runs[runs.kind == "none"]
    runs = pd.concat([runs[runs.kind != "none"]] + [base.assign(kind=k) for k in ERRORS], ignore_index=True)
    runs.to_csv(out / "errors_runs.csv", index=False)
    ref = base.set_index(["rep", "ratio"]).share
    runs["change"] = runs.share - ref.reindex(pd.MultiIndex.from_frame(runs[["rep", "ratio"]])).to_numpy()
    g = runs.groupby(["ratio", "kind", "size"])
    cols = ["share", "change", "used_cov", "used_sev", "theta", "mae_p", "mae_pc"]
    summ = g[cols].mean().join((g[["share", "change", "used_cov", "used_sev"]].std() / np.sqrt(args.reps))
                               .add_prefix("se_"))
    summ["exceed_cov"] = g.used_cov.apply(lambda v: 100 * (v > 100).mean())
    summ["exceed_sev"] = g.used_sev.apply(lambda v: 100 * (v > 100).mean())
    summ = summ.reset_index()
    summ.to_csv(out / "errors_summary.csv", index=False)

    pd.set_option("display.width", 220, "display.max_rows", 200)
    for ratio, a_sev, th_star, best in targets:
        s = summ[summ.ratio == ratio]
        print(f"\nalpha_sev = {ratio:g} S_sev (theta* = {th_star:.3f}, optimal share {100 * best:.2f}%). Change in "
              f"share passed against size 0 (pp, paired), mean risk / level x 100, % of repetitions above the level:")
        print(s.set_index(["kind", "size"])[["share", "change", "se_change", "used_cov", "used_sev", "exceed_cov",
                                             "exceed_sev", "theta", "mae_p", "mae_pc"]].round(3).to_string())
    print("(CRC bounds the mean risk over repetitions; single repetitions above the level are expected)")

    # figure at the first level: (a) bias, (b) noise. Change in share passed against no error
    s = summ[summ.ratio == args.ratios[0]]
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    for ax, how, letter in zip(axes, ("bias", "noise"), "ab"):
        for which in ("p", "c"):
            d = s[s.kind == f"{which} {how}"].sort_values("size")
            ax.errorbar(d["size"], d["change"], yerr=2 * d["se_change"], color=ERR_COL[which], marker="o", ms=2.5,
                        lw=1.3, capsize=0, label=f"error in {which}")
        ax.axhline(0, color=MUTED, lw=0.8, ls=":")
        if how == "bias":
            ax.axvline(0, color=MUTED, lw=0.8, ls=":")
        xl = ("bias on the logit scale (< 0 under-, > 0 over-estimate)" if how == "bias"
              else "noise SD on the logit scale")
        ax.set_xlabel(f"{xl}\n({letter})", fontsize=8, color=INK, linespacing=1.7)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.tick_params(colors=INK, labelsize=7)
    axes[0].set_ylabel("change in share passed (pp)", fontsize=8, color=INK)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_errors.png", dpi=300)
    print(f"\nSaved errors_runs.csv, errors_summary.csv and fig_errors.png to {out}  [{time.time() - t0:.0f}s]")


# ---------------------------------------------------------------- command line
def cli():
    ap = argparse.ArgumentParser(description="Simulation study (Section 5, Appendix C)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("fig1a", help="pass regions in the (p, pc) plane")
    a.add_argument("--ratio", type=float, default=0.5, help="alpha_sev as a multiple of S_sev")
    a.add_argument("--n", type=int, default=400_000)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--no-diagonal", action="store_true", help="do not draw the diagonal pc = p")

    e = sub.add_parser("levels", help="two risks: positive prices, gain and levels used per alpha_sev")
    e.add_argument("--n", type=int, default=400_000)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--levels", type=int, default=40)
    e.add_argument("--step", type=float, default=0.001, help="step of the grid of theta")
    e.add_argument("--bisect", type=int, default=30, help="bisection steps for theta* where both prices are positive")

    t = sub.add_parser("three", help="three risks")
    t.add_argument("--parts", nargs="+", default=["map", "cal"], choices=["map", "mapfig", "cal", "labels"],
                   help="mapfig redraws fig_three_map.png from three_map.csv")
    t.add_argument("--n", type=int, default=200_000, help="cases for the map")
    t.add_argument("--grid", type=int, default=12, help="levels per axis of the map")
    t.add_argument("--lo", type=float, default=0.15, help="smallest level, as a multiple of S_sev and S_late")
    t.add_argument("--hi", type=float, default=1.5, help="largest level, as a multiple of S_sev and S_late")
    t.add_argument("--steps", type=int, default=1000, help="grid steps of the weight of the drop-one scores")
    t.add_argument("--cells", nargs="+", default=["0.5,0.5", "0.5,1.5", "1.5,1.5"],
                   help="cells of the cal part, 'alpha_sev / S_sev,alpha_late / S_late'. The first one is drawn")
    t.add_argument("--label-cells", nargs="+", default=["0.5,0.5", "0.5,1.5"], help="cells of the labels part")
    t.add_argument("--budgets", type=int, nargs="+", default=[1_000, 2_000, 5_000, 10_000, 20_000, 40_000],
                   help="labeled cases m in the labels part")
    t.add_argument("--wstep", type=float, default=0.05, help="grid step of the weights chosen on labels")
    t.add_argument("--reps", type=int, default=100)
    t.add_argument("--n-sample", type=int, default=40_000, help="size of the training, U and calibration samples")
    t.add_argument("--ref", type=int, default=400_000, help="cases for the true prices")
    t.add_argument("--eval", type=int, default=400_000, help="cases with the true risks for evaluation")
    t.add_argument("--seed", type=int, default=0)

    r = sub.add_parser("errors", help="two risks: value of better estimates")
    r.add_argument("--ratios", type=float, nargs="+", default=[0.5], help="alpha_sev as multiples of S_sev")
    r.add_argument("--biases", type=float, nargs="+", default=[-1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1],
                   help="biases on the logit scale")
    r.add_argument("--noises", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0],
                   help="noise standard deviations on the logit scale")
    r.add_argument("--reps", type=int, default=100)
    r.add_argument("--n-u", type=int, default=20_000)
    r.add_argument("--n-cal", type=int, default=20_000)
    r.add_argument("--big", type=int, default=1_000_000, help="cases for alpha_sev and theta*")
    r.add_argument("--eval", type=int, default=400_000)
    r.add_argument("--seed", type=int, default=0)

    c = sub.add_parser("fig2", help="sampling error of the weight and the full method")
    c.add_argument("--parts", nargs="+", default=["a", "c"], choices=["a", "c"])
    c.add_argument("--plot-only", action="store_true", help="redraw the figure from the saved results")
    c.add_argument("--reps-a", type=int, default=100)
    c.add_argument("--reps-c", type=int, default=20)
    c.add_argument("--big", type=int, default=1_000_000, help="size of the large sample for S_sev")
    c.add_argument("--ref", type=int, default=4_000_000, help="size of the sample for the reference theta*")
    c.add_argument("--eval", type=int, default=1_000_000, help="size of the evaluation sample in part c")
    c.add_argument("--seed", type=int, default=0)

    for p_ in (a, e, t, r, c):
        p_.add_argument("--out", default=OUT)
    args = ap.parse_args()
    {"fig1a": fig1a, "levels": levels_fig, "three": three, "errors": errors, "fig2": fig2}[args.cmd](args)


if __name__ == "__main__":
    cli()