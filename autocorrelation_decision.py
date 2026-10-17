"""
Serial-autocorrelation diagnostics and MK / MMK decision rule (one CSV per temporal scale).

Input : CSV with 'Year' + one column per station (e.g. Seasonal_PET_FMAM.csv)
Output: Autocorr_<name>.csv  (Table-2-ready values + decision) and a printed summary

Procedure (matches the wording proposed for Section 3.3.1 / 3.3.2)
1. Ljung-Box Q test on the first LB_LAGS autocorrelation coefficients of the series
   (H0: no autocorrelation).  Criterion for the decision rule.
2. Durbin-Watson statistic on the residuals of a linear regression of the series on time
   (complementary lag-1 diagnostic).  p-values by Monte Carlo under independent normal errors.
   NOTE: DW must NOT be computed on the first-differenced series (white noise would give DW ~ 3).
3. MK test (tie-corrected variance, continuity-corrected Z).
4. MMK test = Hamed & Rao (1998): Var*(S) = Var(S) * n/n*, where the autocorrelations are those of the
   ranks of the Sen-slope-detrended series, using every lag k = 1..n-1 that is significant at 5%
   (|rho_k| > 1.96/sqrt(n)).  (This reproduces the MMK p-values reported in Tables 3-6.)
5. Decision rule:  Ljung-Box p < ALPHA_LB  ->  MMK p-value is used;  otherwise MK p-value is used.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats

# ---------------- SETTINGS ----------------
INPUT_CSV = "Seasonal_PET_FMAM.csv"
N_USED = 30            # years used (30 reproduces Tables 3-6; Table 2 as submitted used 31)
LB_LAGS = 10
ALPHA_LB = 0.05
OUT_DIR = "results"
DW_SIMS = 20000
SEED = 1
# ------------------------------------------


def acf(v, k):
    m = v.mean()
    return np.sum((v[:len(v) - k] - m) * (v[k:] - m)) / np.sum((v - m) ** 2)


def ljung_box(v, lags):
    n = len(v)
    r = np.array([acf(v, k) for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return q, 1 - stats.chi2.cdf(q, lags)


def dw_stat(e):
    return np.sum(np.diff(e) ** 2) / np.sum(e ** 2)


def dw_test(v, rng):
    """DW on residuals of OLS(series ~ time); Monte Carlo two-sided p-value."""
    n = len(v)
    t = np.arange(n)
    X = np.column_stack([np.ones(n), t])
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    M = np.eye(n) - H
    obs = dw_stat(M @ v)
    E = rng.standard_normal((DW_SIMS, n)) @ M.T          # residuals under iid normal errors
    sim = np.sum(np.diff(E, axis=1) ** 2, axis=1) / np.sum(E ** 2, axis=1)
    p_pos = np.mean(sim <= obs)
    p_neg = np.mean(sim >= obs)
    return obs, min(1.0, 2 * min(p_pos, p_neg)), int(p_pos < 0.025) - int(p_neg < 0.025)


def mk_parts(v):
    n = len(v)
    i, j = np.triu_indices(n, 1)
    s = np.sign(v[j] - v[i]).sum()
    _, c = np.unique(v, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(c * (c - 1) * (2 * c + 5))) / 18.0
    return s, var


def p_from(s, var):
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0
    return 2 * (1 - stats.norm.cdf(abs(z)))


def sen_slope(v):
    n = len(v)
    i, j = np.triu_indices(n, 1)
    return np.median((v[j] - v[i]) / (j - i))


def mmk_p(v):
    n = len(v)
    s, var = mk_parts(v)
    r = stats.rankdata(v - sen_slope(v) * np.arange(n))
    tot = 0.0
    for k in range(1, n):
        rk = acf(r, k)
        if abs(rk) > 1.96 / np.sqrt(n):
            tot += (n - k) * (n - k - 1) * (n - k - 2) * rk
    fac = 1 + 2 / (n * (n - 1) * (n - 2)) * tot
    if fac <= 0:          # negative correction factor is undefined -> fall back to MK (fac = 1)
        fac = 1.0
    return p_from(s, var * fac), fac


def main():
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(INPUT_CSV)
    stations = [c for c in df.columns if c.lower() != "year"]
    rows = []
    for st in stations:
        v = df[st].dropna().to_numpy(float)[:N_USED]
        q, p_lb = ljung_box(v, LB_LAGS)
        dw, p_dw, dw_dir = dw_test(v, rng)
        s, var = mk_parts(v)
        p_mk = p_from(s, var)
        p_mmk, fac = mmk_p(v)
        ac = p_lb < ALPHA_LB
        rows.append({"Station": st, "n": len(v), "DW": dw, "DW_p": p_dw,
                     "DW_dir": {1: "positive", -1: "negative", 0: "none"}[int(dw_dir)],
                     "lag1_r": acf(v, 1), "LB_Q": q, "LB_p": p_lb, "Autocorrelated": ac,
                     "MK_p": p_mk, "MMK_p": p_mmk, "n/n*": fac,
                     "Test_used": "MMK" if ac else "MK", "Rule_p": p_mmk if ac else p_mk})
    out = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    name = os.path.splitext(os.path.basename(INPUT_CSV))[0]
    out.to_csv(os.path.join(OUT_DIR, f"Autocorr_{name}.csv"), index=False)

    n_ac = int(out.Autocorrelated.sum())
    agree = int(((out.MK_p < .05) == (out.MMK_p < .05)).sum())
    print(f"{name}: {len(out)} series, n = {N_USED}")
    print(f"  Ljung-Box (lag {LB_LAGS}) p<{ALPHA_LB}: {n_ac} series -> MMK; {len(out) - n_ac} -> MK")
    print(f"  Durbin-Watson (residuals): positive {int((out.DW_dir=='positive').sum())}, "
          f"negative {int((out.DW_dir=='negative').sum())}, none {int((out.DW_dir=='none').sum())}")
    print(f"  MK and MMK agree on 5% significance: {agree}/{len(out)}")
    print(f"  Significant at 5% with the selected test: {int((out.Rule_p < .05).sum())}")


if __name__ == "__main__":
    main()
