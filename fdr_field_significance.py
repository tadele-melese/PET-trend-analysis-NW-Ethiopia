"""
Multiple-testing control and field significance for station trend tests
(MK and ITA), answering the reviewer's comment.

Input : CSV with 'Year' + one column per station (same file as ita_analysis.py)
Output: FDR_FieldSignificance_<name>.csv  and a printed summary

What it does
1. Mann-Kendall (MK) test per station            -> p-value
2. ITA slope S (Sen 2012) per station            -> (a) Sen's analytical p-value
                                                    (b) permutation p-value (calibrated)
3. Benjamini-Hochberg (BH) FDR and Benjamini-Yekutieli (BY, valid under
   dependence) FDR across the stations
4. Field significance (Livezey & Chen 1983 / Wilks 2016): is the NUMBER of
   locally significant stations larger than expected by chance, given the
   spatial correlation among stations?  (years are resampled jointly for all
   stations, which keeps the spatial correlation intact)
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------- SETTINGS ----------------
INPUT_CSV = "Seasonal_PET_FMAM.csv"
N_USED = 30            # same as ITA analysis (first 30 years -> halves of 15)
ALPHA = 0.05
OUT_DIR = "results"
B = 5000               # number of permutations
SEED = 42
# ------------------------------------------


def mk_stat(X):
    """Mann-Kendall S statistic for each column of X (rows = time)."""
    n = X.shape[0]
    i, j = np.triu_indices(n, k=1)
    return np.sign(X[j, :] - X[i, :]).sum(axis=0)


def mk_pvalue(x):
    n = len(x)
    s = mk_stat(x[:, None])[0]
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return 2 * (1 - stats.norm.cdf(abs(z)))


def ita_S(X):
    """ITA slope per column (does not depend on the sorting)."""
    N = X.shape[0]
    n = N // 2
    return 2.0 * (X[n:, :].mean(axis=0) - X[:n, :].mean(axis=0)) / N


def ita_sen_p(x):
    """Sen (2012) analytical two-sided p-value of the ITA slope."""
    N = len(x)
    n = N // 2
    a, b = np.sort(x[:n]), np.sort(x[n:])
    rho = np.corrcoef(a, b)[0, 1]
    sig_s = 2 * np.sqrt(2) * x.std(ddof=1) * np.sqrt(1 - rho) / (N * np.sqrt(N))
    S = 2 * (b.mean() - a.mean()) / N
    return 2 * (1 - stats.norm.cdf(abs(S) / sig_s))


def fdr(p, q=0.05, dependent=False):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    c = np.sum(1 / np.arange(1, m + 1)) if dependent else 1.0
    thr = q * np.arange(1, m + 1) / (m * c)
    ok = p[order] <= thr
    k = np.max(np.where(ok)[0]) + 1 if ok.any() else 0
    rej = np.zeros(m, bool)
    rej[order[:k]] = True
    return rej


def main():
    rng = np.random.default_rng(SEED)      # re-seeded for every file -> reproducible
    df = pd.read_csv(INPUT_CSV)
    stations = [c for c in df.columns if c.lower() != "year"]
    X = df[stations].to_numpy(float)[:N_USED, :]
    N, m = X.shape

    # ---- observed statistics -------------------------------------------
    mk_p = np.array([mk_pvalue(X[:, k]) for k in range(m)])
    S_obs = ita_S(X)
    ita_p_sen = np.array([ita_sen_p(X[:, k]) for k in range(m)])

    # ---- permutation null (years shuffled jointly for all stations) ----
    absS_null = np.empty((B, m))
    absMK_null = np.empty((B, m))
    for b in range(B):
        Xp = X[rng.permutation(N), :]
        absS_null[b] = np.abs(ita_S(Xp))
        absMK_null[b] = np.abs(mk_stat(Xp))
    absMK_obs = np.abs(mk_stat(X))

    # per-station permutation p-values (observed vs its own null)
    ita_p_perm = (1 + (absS_null >= np.abs(S_obs)).sum(axis=0)) / (B + 1)
    mk_p_perm = (1 + (absMK_null >= absMK_obs).sum(axis=0)) / (B + 1)

    # ---- FDR ------------------------------------------------------------
    out = pd.DataFrame({"Station": stations, "MK_p": mk_p, "ITA_S": S_obs,
                        "ITA_p_Sen": ita_p_sen, "ITA_p_perm": ita_p_perm})
    for col in ["MK_p", "ITA_p_Sen", "ITA_p_perm"]:
        out[f"{col}_sig"] = out[col] < ALPHA
        out[f"{col}_BH"] = fdr(out[col].values, ALPHA)
        out[f"{col}_BY"] = fdr(out[col].values, ALPHA, dependent=True)

    # ---- field significance --------------------------------------------
    # null distribution of the number of locally significant stations:
    # every permutation is treated as "observed" and ranked against the others
    def field_test(abs_null, abs_obs):
        # p of each permutation at each station (rank-based)
        ranks = (-abs_null).argsort(axis=0).argsort(axis=0)   # 0 = largest
        p_perm = (ranks + 1) / B
        n_sig_null = (p_perm < ALPHA).sum(axis=1)
        n_sig_obs = ((1 + (abs_null >= abs_obs).sum(axis=0)) / (B + 1) < ALPHA).sum()
        p_field = (1 + (n_sig_null >= n_sig_obs).sum()) / (B + 1)
        return n_sig_obs, np.percentile(n_sig_null, 95), p_field

    fs_mk = field_test(absMK_null, absMK_obs)
    fs_ita = field_test(absS_null, np.abs(S_obs))

    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(INPUT_CSV))[0]
    out.to_csv(os.path.join(OUT_DIR, f"FDR_FieldSignificance_{name}.csv"), index=False)

    print(f"File: {INPUT_CSV}")
    print(f"Stations: {m}   years used: {N}   alpha = {ALPHA}\n")
    print("Number of significant stations")
    print(f"  MK   raw p<{ALPHA}: {int(out.MK_p_sig.sum()):2d} | BH: {int(out.MK_p_BH.sum()):2d} | BY: {int(out.MK_p_BY.sum()):2d}")
    print(f"  ITA  Sen CL (analytical): {int(out.ITA_p_Sen_sig.sum()):2d} | BH: {int(out.ITA_p_Sen_BH.sum()):2d} | BY: {int(out.ITA_p_Sen_BY.sum()):2d}")
    print(f"  ITA  permutation        : {int(out.ITA_p_perm_sig.sum()):2d} | BH: {int(out.ITA_p_perm_BH.sum()):2d} | BY: {int(out.ITA_p_perm_BY.sum()):2d}")
    print("\nField significance (permutation, spatial correlation preserved)")
    print(f"  MK : observed {fs_mk[0]} sig. stations, 95th pct of null = {fs_mk[1]:.0f}, field p = {fs_mk[2]:.4f}")
    print(f"  ITA: observed {fs_ita[0]} sig. stations, 95th pct of null = {fs_ita[1]:.0f}, field p = {fs_ita[2]:.4f}")


if __name__ == "__main__":
    main()
