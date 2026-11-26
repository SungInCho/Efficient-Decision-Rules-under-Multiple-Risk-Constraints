"""
Data splits for the LendingClub experiments.

Loans issued before 2015 (pre-2015) are split into three disjoint sets (70 / 10 / 20 of pre-2015):
  hold  : drawn at random from the most recent pre-2015 loans (issued in HOLD_YEAR = 2014).
          Never used in training. Outcomes of the most recent past customers, for calibration
          in the deployment design (theta from 2015 applicants without outcomes, cut-off from hold).
  val   : drawn at random from the remaining pre-2015 loans. Early stopping of the three
          models only (its outcomes are used in training).
  train : the rest. Fits p_hat, c1_hat, c0_hat                       (train_models.py)
Loans issued in 2015 form the pool (new customers). split_pool splits it at random.

Because val outcomes choose the number of trees, p_hat and c_hat depend on val.
Calibrating on val would break the condition that the scores do not depend on the
calibration sample. hold has no such dependence.

Usage (repo root):
    python src/splits.py        prints sizes, default rates, date ranges and checks
"""
import numpy as np
import pandas as pd
from pathlib import Path

TRAIN_END = pd.Timestamp("2015-01-01")   # pre-2015 = issue_d before this date
VAL_FRAC = 0.10                          # share of pre-2015 loans for early stopping (as before)
HOLD_FRAC = 0.20                         # share of pre-2015 loans held out for calibration
HOLD_YEAR = 2014                         # hold is drawn only from loans issued in this year
SEED = 42


def load_splits(df, return_hold=False):
    """Return (train, val, pool), or (train, val, hold, pool) with return_hold=True.

    hold is always removed from train, so every caller sees the same training set.
    """
    pre = df[df["issue_d"] < TRAIN_END]
    pool_df = df[df["issue_d"] >= TRAIN_END].reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    n_val, n_hold = int(round(VAL_FRAC * len(pre))), int(round(HOLD_FRAC * len(pre)))
    recent = np.flatnonzero((pre["issue_d"].dt.year == HOLD_YEAR).to_numpy())
    assert n_hold <= len(recent), f"hold needs {n_hold:,} loans but {HOLD_YEAR} has {len(recent):,}"
    hold_pos = rng.choice(recent, size=n_hold, replace=False)
    rest = np.setdiff1d(np.arange(len(pre)), hold_pos)
    rest = rng.permutation(rest)
    hold_df = pre.iloc[np.sort(hold_pos)]
    val_df = pre.iloc[rest[:n_val]]
    train_df = pre.iloc[rest[n_val:]]

    if return_hold:
        return train_df, val_df, hold_df, pool_df
    return train_df, val_df, pool_df


def split_pool(n_pool, seed, cal_frac=0.5):
    """Random split of the 2015 pool indices into two parts (unchanged)."""
    perm = np.random.default_rng(seed).permutation(n_pool)
    k = int(n_pool * cal_frac)
    return perm[:k], perm[k:]


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "lc36_2012_2015.parquet"
    df = pd.read_parquet(data_path)

    train_df, val_df, hold_df, pool_df = load_splits(df, return_hold=True)
    pre_all = df[df["issue_d"] < TRAIN_END]
    cal_idx, test_idx = split_pool(len(pool_df), seed=0)
    cal_df, test_df = pool_df.iloc[cal_idx], pool_df.iloc[test_idx]
    sets = {"train": train_df, "val": val_df, "hold": hold_df, "pool": pool_df,
            "cal": cal_df, "test": test_df}

    print("size / default rate / issue_d range")
    for name, d in sets.items():
        print(f"  {name:5s} n={len(d):>7,}  default={d['y'].mean():.4f}  "
              f"issue_d {d['issue_d'].min():%Y-%m} ~ {d['issue_d'].max():%Y-%m}")
    assert len(train_df) + len(val_df) + len(hold_df) + len(pool_df) == len(df)
    assert len(cal_idx) + len(test_idx) == len(pool_df)

    print("\nid overlap (all should be 0)")
    names = ["train", "val", "hold", "cal", "test"]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n = len(set(sets[names[i]]["id"]) & set(sets[names[j]]["id"]))
            print(f"  {names[i]} & {names[j]}: {n}")
            assert n == 0

    print("\nreproducibility")
    a = load_splits(df, return_hold=True)
    print(f"  same split twice -> identical: {all(x.index.equals(y.index) for x, y in zip(a, (train_df, val_df, hold_df, pool_df)))}")
    c2, t2 = split_pool(len(pool_df), seed=0)
    c3, _ = split_pool(len(pool_df), seed=1)
    print(f"  pool, same seed -> identical: {np.array_equal(cal_idx, c2) and np.array_equal(test_idx, t2)}")
    print(f"  pool, different seed -> different: {not np.array_equal(cal_idx, c3)}")

    print(f"\nshares of pre-2015: train {len(train_df)/len(pre_all):.3f}  val {len(val_df)/len(pre_all):.3f}  hold {len(hold_df)/len(pre_all):.3f}")
    print(f"share of {HOLD_YEAR} loans left in train: {(train_df['issue_d'].dt.year == HOLD_YEAR).sum() / (pre_all['issue_d'].dt.year == HOLD_YEAR).sum():.3f}")
    print("\ndefault-rate gaps")
    print(f"  hold - train: {hold_df['y'].mean() - train_df['y'].mean():+.4f}")
    print(f"  hold - val  : {hold_df['y'].mean() - val_df['y'].mean():+.4f}")
    for s in range(5):
        c, t = split_pool(len(pool_df), seed=s)
        print(f"  pool seed {s}, cal - test: {pool_df['y'].iloc[c].mean() - pool_df['y'].iloc[t].mean():+.4f}")