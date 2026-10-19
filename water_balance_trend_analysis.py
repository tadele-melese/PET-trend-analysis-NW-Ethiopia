#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Water-balance trend analysis (Annual, JJAS, FMAM, ONDJ) for multiple stations
=============================================================================

Input  : Annual_water_balance.csv, JJAS_water_balance.csv,
         FMAM_water_balance.csv, ONDJ_water_balance.csv
         (first column = year, remaining columns = stations)
Methods:
  1. Mann-Kendall (MK) test, tie-corrected variance, continuity-corrected Z
  2. Modified Mann-Kendall (MMK), Hamed & Rao (1998) variance correction for
     serial correlation (autocorrelation of ranks of the detrended series)
  3. Theil-Sen slope with 95 % confidence interval (per year / per decade)
  4. Innovative Trend Analysis (ITA), Sen (2012, 2017):
       slope s = 2 (mean(2nd half) - mean(1st half)) / n
       sigma_s = 2*sqrt(2)*sigma / (n*sqrt(pi)) * sqrt(1 - rho)
       Sen (2017) limits: significant if |s| > z(1-alpha/2) * sigma_s (kept for
       comparability; liberal) AND a calibrated permutation p-value of s
       (default decision rule, two-sided, same alpha as MK/MMK)
       D = 10 * mean[(y_i - x_i)/mean(x)]  (% indicator, Sen 2017)
  5. Multiple-testing control across the stations of each season:
       - Benjamini-Hochberg false discovery rate (FDR) adjusted p-values
       - Field significance by year-resampling (Livezey & Chen, 1983), which
         keeps the spatial correlation among stations
  6. Regional (all-station mean) series tested with the same methods
  7. Data QC report and a sensitivity run with user-defined suspect values
     masked (see SENSITIVITY_MASK)

Outputs (in OUTPUT_DIR): Excel workbook, CSV, PNG/PDF figures, methods note.

Usage
-----
    python water_balance_trend_analysis.py
    python water_balance_trend_analysis.py --input "D:\\Collegues\\Abebaw\\Water Balance"
    python water_balance_trend_analysis.py --input <dir> --output <dir> --alpha 0.05

Requires: numpy, pandas, scipy, matplotlib, openpyxl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

# =============================================================================
# 1. CONFIGURATION  (edit here only)
# =============================================================================
INPUT_DIR = Path(r"D:\Collegues\Abebaw\Water Balance")
OUTPUT_DIR = None                      # None -> <INPUT_DIR>/trend_results

SEASON_FILES = {                        # order = order in tables and figures
    "Annual": "Annual_water_balance.csv",
    "JJAS": "JJAS_water_balance.csv",   # Kiremt
    "FMAM": "FMAM_water_balance.csv",   # Belg
    "ONDJ": "ONDJ_water_balance.csv",   # Bega
}
SEASON_LABELS = {
    "Annual": "Annual",
    "JJAS": "JJAS (Kiremt)",
    "FMAM": "FMAM (Belg)",
    "ONDJ": "ONDJ (Bega)",
}

VARIABLE_LABEL = "Water balance"        # axis / caption label
UNIT = "%"                              # <-- confirm the unit of your data

ALPHA = 0.05                            # significance level (two-sided)
ITA_LEVELS = (0.10, 0.05, 0.01)         # CL90 / CL95 / CL99 (as in earlier ITA script)
ITA_D_FACTOR = 10                       # D = 10 * mean[(y-x)/mean(x)] (Sen 2017)
# Decision rule for ITA significance:
#   "permutation" -> exact resampling p-value of the ITA slope (calibrated; default)
#   "sen2017"     -> analytic limits of Sen (2017) (as in the earlier ITA script;
#                    known to be liberal: ~35 % false positives at nominal 5 % for n = 31)
ITA_CRITERION = "permutation"
N_ITA_PERM = 5000
N_PERMUTATIONS = 5000                   # field-significance resampling
RANDOM_SEED = 2026
MIN_OBS = 10                            # minimum valid years per series

# Sensitivity run: values replaced by NaN before re-running all tests.
# Default = the implausible start-of-record values found in the QC of the
# supplied files (e.g. AbaySheleko 0.00 in 1984-85 and Bure 71/100 in 1984-85
# against a normal level of ~25). Edit or empty ({}) after checking raw data.
SENSITIVITY_MASK = {
    "AbaySheleko": [1984, 1985],
    "Bure": [1984, 1985],
}

DPI = 300
FONT = {"family": "sans-serif", "sans-serif": ["Arial", "DejaVu Sans"], "size": 8}
C_INC, C_DEC, C_NS = "#2166ac", "#b2182b", "#9e9e9e"   # colour-blind safe

mpl.rc("font", **FONT)
mpl.rcParams.update({
    "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": DPI,
})

log = logging.getLogger("wb_trend")


# =============================================================================
# 2. STATISTICAL FUNCTIONS
# =============================================================================
def _tie_term(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    return float(np.sum(counts * (counts - 1) * (2 * counts + 5)))


def mk_statistic(x: np.ndarray) -> float:
    """Mann-Kendall S."""
    iu = np.triu_indices(len(x), 1)
    return float(np.sign(x[iu[1]] - x[iu[0]]).sum())


def mk_var(x: np.ndarray) -> float:
    n = len(x)
    return (n * (n - 1) * (2 * n + 5) - _tie_term(x)) / 18.0


def z_from_s(s: float, var: float) -> float:
    if var <= 0:
        return np.nan
    if s > 0:
        return (s - 1) / np.sqrt(var)
    if s < 0:
        return (s + 1) / np.sqrt(var)
    return 0.0


def sen_slope(t: np.ndarray, x: np.ndarray):
    """Theil-Sen slope and intercept (median of pairwise slopes)."""
    iu = np.triu_indices(len(x), 1)
    dt = t[iu[1]] - t[iu[0]]
    slopes = (x[iu[1]] - x[iu[0]]) / dt
    b = np.median(slopes)
    a = np.median(x - b * t)
    return b, a, np.sort(slopes)


def sen_ci(sorted_slopes: np.ndarray, var_s: float, alpha: float = 0.05):
    """Sen (1968) confidence interval for the slope, rank interpolation."""
    N = len(sorted_slopes)
    c = stats.norm.ppf(1 - alpha / 2) * np.sqrt(var_s)
    ranks = np.arange(1, N + 1)
    lo = np.interp((N - c) / 2.0, ranks, sorted_slopes)
    hi = np.interp((N + c) / 2.0 + 1.0, ranks, sorted_slopes)
    return lo, hi


def hamed_rao_factor(t: np.ndarray, x: np.ndarray, slope: float) -> float:
    """n/n* of Hamed & Rao (1998): variance inflation from significant lags."""
    n = len(x)
    detr = x - slope * t
    ranks = stats.rankdata(detr)
    r = ranks - ranks.mean()
    denom = np.sum(r * r)
    if denom == 0:
        return 1.0
    acf = np.array([np.sum(r[:n - k] * r[k:]) / denom for k in range(1, n)])
    k = np.arange(1, n)
    z = stats.norm.ppf(0.975)
    lower = (-1 - z * np.sqrt(n - k - 1)) / (n - k)
    upper = (-1 + z * np.sqrt(n - k - 1)) / (n - k)
    sig = (acf < lower) | (acf > upper)
    if not sig.any():
        return 1.0
    kk = k[sig]
    corr = np.sum((n - kk) * (n - kk - 1) * (n - kk - 2) * acf[sig])
    return float(1 + 2.0 / (n * (n - 1) * (n - 2)) * corr)


_PERM_CACHE: dict = {}


def _perm_matrix(n: int) -> np.ndarray:
    if n not in _PERM_CACHE:
        rng = np.random.default_rng(RANDOM_SEED + n)
        _PERM_CACHE[n] = np.argsort(rng.random((N_ITA_PERM, n)), axis=1)
    return _PERM_CACHE[n]


def ita_test(x: np.ndarray, levels=ITA_LEVELS) -> dict:
    """Innovative Trend Analysis with Sen (2017) significance limits."""
    x = np.asarray(x, float)
    if len(x) % 2:                       # odd length -> drop the middle year
        x = np.delete(x, len(x) // 2)
    n = len(x)
    m = n // 2
    a, b = np.sort(x[:m]), np.sort(x[m:])
    s = 2.0 * (b.mean() - a.mean()) / n
    sd = x.std(ddof=1)
    rho = np.corrcoef(a, b)[0, 1] if (a.std() > 0 and b.std() > 0) else np.nan
    sigma_s = (2 * np.sqrt(2) * sd / (n * np.sqrt(np.pi)) * np.sqrt(max(1 - rho, 0))
               if np.isfinite(rho) else np.nan)
    z = s / sigma_s if sigma_s and sigma_s > 0 else np.nan
    out = {
        "ITA_n_used": n,
        "ITA_slope": s,
        "ITA_rho": rho,
        "ITA_sigma_s": sigma_s,
        "ITA_Z_Sen2017": z,
        "ITA_p_Sen2017": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan,
        "ITA_D_pct": ITA_D_FACTOR * np.mean((b - a) / a.mean()) if a.mean() != 0 else np.nan,
    }
    for lv in levels:
        out[f"ITA_CL{int(round((1 - lv) * 100))}"] = (
            stats.norm.ppf(1 - lv / 2) * sigma_s if np.isfinite(sigma_s) else np.nan)
    sig_level = "NS"
    for lv in sorted(levels, reverse=True):          # 0.10 -> 0.01
        cl = out[f"ITA_CL{int(round((1 - lv) * 100))}"]
        if np.isfinite(cl) and abs(s) > cl:
            sig_level = f"{int(round((1 - lv) * 100))}%"
    out["ITA_sig_level_Sen2017"] = sig_level
    # calibrated alternative: permutation distribution of the ITA slope under
    # exchangeable years (the sorting inside the halves does not change s)
    xs = x[_perm_matrix(n)]
    s_null = 2.0 * (xs[:, m:].mean(axis=1) - xs[:, :m].mean(axis=1)) / n
    out["ITA_p_perm"] = float((np.sum(np.abs(s_null) >= abs(s) - 1e-12) + 1) / (N_ITA_PERM + 1))
    out["ITA_p"] = out["ITA_p_perm"] if ITA_CRITERION == "permutation" else out["ITA_p_Sen2017"]
    out["ITA_first_half_sorted"] = a
    out["ITA_second_half_sorted"] = b
    return out


def trend_label(z: float, p: float, alpha: float) -> str:
    if not np.isfinite(p) or p >= alpha:
        return "No trend"
    return "Increasing" if z > 0 else "Decreasing"


def analyse_series(years, values, alpha=ALPHA) -> dict | None:
    """All tests for one series (NaNs are dropped, years kept as time axis)."""
    t = np.asarray(years, float)
    x = np.asarray(values, float)
    ok = np.isfinite(x)
    t, x = t[ok], x[ok]
    n = len(x)
    if n < MIN_OBS or np.nanstd(x) == 0:
        return None

    s = mk_statistic(x)
    var = mk_var(x)
    z = z_from_s(s, var)
    p = 2 * stats.norm.sf(abs(z))
    tau = s / (0.5 * n * (n - 1))

    b, a, sorted_slopes = sen_slope(t, x)
    factor = hamed_rao_factor(t, x, b)
    var_mmk = var * factor
    z_m = z_from_s(s, var_mmk)
    p_m = 2 * stats.norm.sf(abs(z_m))
    lo, hi = sen_ci(sorted_slopes, var_mmk, alpha)
    mean = x.mean()

    res = {
        "n_years": n, "start_year": int(t.min()), "end_year": int(t.max()),
        "mean": mean, "std": x.std(ddof=1),
        "MK_S": s, "MK_tau": tau, "MK_Z": z, "MK_p": p,
        "MK_trend": trend_label(z, p, alpha),
        "MMK_nn_ratio": factor, "MMK_Z": z_m, "MMK_p": p_m,
        "MMK_trend": trend_label(z_m, p_m, alpha),
        "Sen_slope_per_yr": b,
        "Sen_slope_per_decade": b * 10,
        "Sen_CI_low_per_decade": lo * 10,
        "Sen_CI_high_per_decade": hi * 10,
        "Sen_change_pct_of_mean_per_decade": 100 * b * 10 / mean if mean else np.nan,
        "Sen_total_change": b * (t.max() - t.min()),
    }
    ita = ita_test(x)
    res.update({k: v for k, v in ita.items() if not k.endswith("sorted")})
    res["ITA_trend"] = (("Increasing" if ita["ITA_slope"] > 0 else "Decreasing")
                        if np.isfinite(ita["ITA_p"]) and ita["ITA_p"] < alpha else "No trend")
    pi = ita["ITA_p"]
    res["ITA_sig_level"] = ("99%" if pi < 0.01 else "95%" if pi < 0.05 else
                            "90%" if pi < 0.10 else "NS") if np.isfinite(pi) else "NS"
    res["_ita_a"], res["_ita_b"] = ita["ITA_first_half_sorted"], ita["ITA_second_half_sorted"]
    res["_t"], res["_x"], res["_a"], res["_b"] = t, x, a, b
    return res


# ---- multiple testing -------------------------------------------------------
def fdr_bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    m = len(pv)
    if m == 0:
        return out
    order = np.argsort(pv)
    adj = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    res = np.empty(m)
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def field_significance_mk(data: pd.DataFrame, alpha=ALPHA, n_perm=N_PERMUTATIONS,
                          seed=RANDOM_SEED) -> dict:
    """Livezey-Chen (1983) field significance of the number of locally
    significant MK stations. Years are shuffled jointly across stations, so
    the inter-station correlation is preserved. Requires a complete matrix."""
    X = data.values.astype(float)
    n, k = X.shape
    var = np.array([mk_var(X[:, j]) for j in range(k)])
    zcrit = stats.norm.ppf(1 - alpha / 2)
    iu = np.triu_indices(n, 1)

    def count(M):
        S = np.sign(M[iu[1], :] - M[iu[0], :]).sum(axis=0)
        z = np.where(S > 0, (S - 1), np.where(S < 0, (S + 1), 0.0)) / np.sqrt(var)
        return int(np.sum(np.abs(z) > zcrit))

    obs = count(X)
    rng = np.random.default_rng(seed)
    null = np.array([count(X[rng.permutation(n), :]) for _ in range(n_perm)])
    return {
        "n_stations": k,
        "n_locally_significant_MK": obs,
        "null_mean_count": float(null.mean()),
        "null_95th_pct_count": float(np.percentile(null, 95)),
        "field_p_value": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
        "field_significant": bool((np.sum(null >= obs) + 1) / (n_perm + 1) < alpha),
    }


# =============================================================================
# 3. DATA HANDLING
# =============================================================================
def load_data(input_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    for season, fname in SEASON_FILES.items():
        f = input_dir / fname
        if not f.exists():
            sys.exit(f"ERROR: file not found: {f}")
        df = pd.read_csv(f, index_col=0)
        df.index = pd.to_numeric(df.index, errors="coerce")
        df = df[df.index.notna()]
        df.index = df.index.astype(int)
        df.index.name = "Year"
        df.columns = [str(c).strip() for c in df.columns]
        df = df.apply(pd.to_numeric, errors="coerce").sort_index()
        data[season] = df
        log.info("Loaded %-6s: %d years (%d-%d) x %d stations, %d missing",
                 season, len(df), df.index.min(), df.index.max(), df.shape[1],
                 int(df.isna().sum().sum()))
    return data


def qc_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for season, df in data.items():
        for st in df.columns:
            x = df[st]
            med = x.median()
            mad = (x - med).abs().median()
            rows.append({
                "Season": season, "Station": st, "n_valid": int(x.notna().sum()),
                "n_missing": int(x.isna().sum()),
                "n_zero": int((x == 0).sum()), "n_100": int((x == 100).sum()),
                "n_unique_values": int(x.nunique()),
                "min": x.min(), "median": med, "max": x.max(),
                "first_years_zero_or_100": ", ".join(
                    str(y) for y in x.index[:3] if x[y] in (0, 100)),
            })
    return pd.DataFrame(rows)


def apply_mask(df: dict[str, pd.DataFrame], mask: dict) -> dict[str, pd.DataFrame]:
    out = {}
    for season, d in df.items():
        d2 = d.copy()
        for st, years in mask.items():
            if st in d2.columns:
                d2.loc[d2.index.isin(years), st] = np.nan
        out[season] = d2
    return out


# =============================================================================
# 4. RUN ANALYSIS
# =============================================================================
def run_all(data: dict[str, pd.DataFrame], alpha=ALPHA, with_field=True):
    """Returns (results DataFrame, dict of raw per-series results, field table)."""
    rows, raw, field_rows = [], {}, []
    for season, df in data.items():
        season_rows = []
        for st in df.columns:
            r = analyse_series(df.index.values, df[st].values, alpha)
            if r is None:
                log.warning("Skipped %s / %s (insufficient or constant data)", season, st)
                continue
            raw[(season, st)] = r
            row = {"Season": season, "Station": st}
            row.update({k: v for k, v in r.items() if not k.startswith("_")})
            season_rows.append(row)
        sd = pd.DataFrame(season_rows)
        for m in ("MK", "MMK", "ITA"):          # FDR across stations, per season
            sd[f"{m}_p_FDR"] = fdr_bh(sd[f"{m}_p"].values)
            sd[f"{m}_sig_FDR"] = sd[f"{m}_p_FDR"] < alpha
        rows.append(sd)
        if with_field and not df.isna().any().any():
            fs = field_significance_mk(df, alpha)
            fs["Season"] = season
            field_rows.append(fs)
    res = pd.concat(rows, ignore_index=True)
    field = pd.DataFrame(field_rows)
    if not field.empty:
        field = field[["Season"] + [c for c in field.columns if c != "Season"]]
    return res, raw, field


def regional_mean(data: dict[str, pd.DataFrame], alpha=ALPHA):
    rows, series = [], {}
    for season, df in data.items():
        reg = df.mean(axis=1, skipna=True)
        series[season] = reg
        r = analyse_series(reg.index.values, reg.values, alpha)
        row = {"Season": season, "Series": "All-station mean"}
        row.update({k: v for k, v in r.items() if not k.startswith("_")})
        rows.append(row)
    return pd.DataFrame(rows), series


def summary_counts(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season in res["Season"].unique():
        d = res[res["Season"] == season]
        for m in ("MK", "MMK", "ITA"):
            inc = int((d[f"{m}_trend"] == "Increasing").sum())
            dec = int((d[f"{m}_trend"] == "Decreasing").sum())
            fdr_inc = int(((d[f"{m}_trend"] == "Increasing") & d[f"{m}_sig_FDR"]).sum())
            fdr_dec = int(((d[f"{m}_trend"] == "Decreasing") & d[f"{m}_sig_FDR"]).sum())
            rows.append({
                "Season": season, "Method": m, "n_stations": len(d),
                "Sig_increasing": inc, "Sig_decreasing": dec,
                "No_trend": len(d) - inc - dec,
                "Sig_increasing_FDR": fdr_inc, "Sig_decreasing_FDR": fdr_dec,
                "Pct_stations_significant": round(100 * (inc + dec) / len(d), 1),
            })
    out = pd.DataFrame(rows)
    # MK vs MMK vs ITA agreement of direction/significance class
    agree = []
    for season in res["Season"].unique():
        d = res[res["Season"] == season]
        a = ((d["MK_trend"] == d["MMK_trend"]) & (d["MK_trend"] == d["ITA_trend"])).sum()
        agree.append({"Season": season, "n_stations": len(d),
                      "All_three_methods_agree": int(a),
                      "Pct_agreement": round(100 * a / len(d), 1)})
    return out, pd.DataFrame(agree)


# =============================================================================
# 5. FIGURES
# =============================================================================
def _save(fig, outdir: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    log.info("Saved figure %s", name)


def fig_ita_grid(raw, season, stations, outdir: Path, tag=""):
    ncol = 5
    nrow = int(np.ceil(len(stations) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.6, 1.55 * nrow + 0.6),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, st in zip(axes, stations):
        r = raw.get((season, st))
        if r is None:
            ax.axis("off")
            continue
        a, b = r["_ita_a"], r["_ita_b"]
        lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
        pad = 0.05 * (hi - lo if hi > lo else 1)
        lim = (lo - pad, hi + pad)
        xx = np.array(lim)
        ax.plot(xx, xx, color="k", lw=0.7)
        ax.plot(xx, xx * 1.10, color="k", lw=0.5, ls="--")
        ax.plot(xx, xx * 0.90, color="k", lw=0.5, ls="--")
        col = C_NS
        if r["ITA_trend"] == "Increasing":
            col = C_INC
        elif r["ITA_trend"] == "Decreasing":
            col = C_DEC
        ax.scatter(a, b, s=9, color=col, edgecolor="none", alpha=0.9, zorder=3)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=6, length=2, pad=1)
        star = {"NS": "", "90%": "*", "95%": "**", "99%": "***"}
        ax.set_title(f"{st}{star.get(r['ITA_sig_level'], '')}", fontsize=7, pad=2)
        ax.text(0.04, 0.95, f"s = {r['ITA_slope']:+.2f}\nD = {r['ITA_D_pct']:+.1f}",
                transform=ax.transAxes, va="top", fontsize=5.5)
    for ax in axes[len(stations):]:
        ax.axis("off")
    fig.supxlabel(f"First half of series, sorted ({UNIT})", fontsize=8)
    fig.supylabel(f"Second half of series, sorted ({UNIT})", fontsize=8)
    fig.suptitle(f"ITA of {VARIABLE_LABEL} - {SEASON_LABELS[season]}", fontsize=9)
    fig.text(0.5, -0.012, "Blue/red = significant increase/decrease at the 5 % level "
             "(two-sided ITA test); grey = no trend. Dashed lines = +/-10 %. "
             "* p<0.10, ** p<0.05, *** p<0.01.",
             ha="center", fontsize=6)
    _save(fig, outdir, f"Fig_ITA_{season}{tag}")


def fig_heatmap(res, stations, outdir: Path, method="MMK", tag="", alpha=ALPHA):
    seasons = list(res["Season"].unique())
    slope = pd.DataFrame({s: res[res.Season == s].set_index("Station")["Sen_slope_per_decade"]
                          for s in seasons}).reindex(stations)
    pcol, fcol = f"{method}_p", f"{method}_sig_FDR"
    praw = pd.DataFrame({s: res[res.Season == s].set_index("Station")[pcol] for s in seasons}).reindex(stations)
    pfdr = pd.DataFrame({s: res[res.Season == s].set_index("Station")[fcol] for s in seasons}).reindex(stations)
    vmax = np.nanmax(np.abs(slope.values))
    fig, ax = plt.subplots(figsize=(4.6, 0.22 * len(stations) + 1.4))
    im = ax.imshow(slope.values, cmap="RdBu", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels([SEASON_LABELS[s] for s in seasons], rotation=30, ha="right")
    ax.set_yticks(range(len(stations)))
    ax.set_yticklabels(stations, fontsize=7)
    for i in range(len(stations)):
        for j, s in enumerate(seasons):
            p, f = praw.iloc[i, j], pfdr.iloc[i, j]
            if pd.notna(p) and p < alpha:
                dark = abs(slope.iloc[i, j]) > 0.6 * vmax
                ax.text(j, i, "\u25CF" if f else "\u25CB", ha="center", va="center",
                        fontsize=7, color="w" if dark else "k")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(f"Sen's slope ({UNIT} decade$^{{-1}}$)")
    ax.set_title(f"{VARIABLE_LABEL} trend, significance by {method}", fontsize=9)
    fig.text(0.5, 0.005, f"\u25CF significant after FDR control (q = {alpha});  "
             f"\u25CB significant at p < {alpha} only (unadjusted)",
             ha="center", fontsize=6.5)
    for sp in ax.spines.values():
        sp.set_visible(False)
    _save(fig, outdir, f"Fig_Heatmap_Sen_{method}{tag}")


def fig_agreement(res, outdir: Path, tag=""):
    seasons = list(res["Season"].unique())
    fig, axes = plt.subplots(1, len(seasons), figsize=(7.4, 2.6), sharey=True,
                             constrained_layout=True)
    for ax, season in zip(np.atleast_1d(axes), seasons):
        d = res[res.Season == season]
        for i, m in enumerate(("MK", "MMK", "ITA")):
            inc = (d[f"{m}_trend"] == "Increasing").sum()
            dec = (d[f"{m}_trend"] == "Decreasing").sum()
            ns = len(d) - inc - dec
            ax.bar(i, inc, color=C_INC, width=0.7)
            ax.bar(i, dec, bottom=inc, color=C_DEC, width=0.7)
            ax.bar(i, ns, bottom=inc + dec, color=C_NS, width=0.7)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["MK", "MMK", "ITA"])
        ax.set_title(SEASON_LABELS[season], fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("Number of stations")
    fig.legend(handles=[Patch(color=C_INC, label="Significant increase"),
                        Patch(color=C_DEC, label="Significant decrease"),
                        Patch(color=C_NS, label="No significant trend")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1))
    _save(fig, outdir, f"Fig_Method_agreement{tag}")


def fig_regional(series, reg_table, outdir: Path, tag=""):
    seasons = list(series.keys())
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.6), constrained_layout=True)
    for ax, s in zip(axes.ravel(), seasons):
        y = series[s]
        ax.plot(y.index, y.values, color="k", lw=0.9, marker="o", ms=2.5)
        r = reg_table[reg_table.Season == s].iloc[0]
        t = y.index.values.astype(float)
        b = r["Sen_slope_per_yr"]
        a = np.median(y.values - b * t)
        col = C_INC if r["MMK_trend"] == "Increasing" else C_DEC if r["MMK_trend"] == "Decreasing" else C_NS
        ax.plot(t, a + b * t, color=col, lw=1.6)
        ax.set_title(SEASON_LABELS[s], fontsize=8)
        ax.text(0.03, 0.96,
                f"Sen = {r['Sen_slope_per_decade']:+.2f} {UNIT}/decade\n"
                f"MK p = {r['MK_p']:.3f};  MMK p = {r['MMK_p']:.3f}\n"
                f"ITA s = {r['ITA_slope']:+.3f} ({r['ITA_sig_level']})",
                transform=ax.transAxes, va="top", fontsize=6.5,
                bbox=dict(fc="white", ec="none", alpha=0.8))
        ax.set_ylabel(f"{VARIABLE_LABEL} ({UNIT})")
    for ax in axes[1, :]:
        ax.set_xlabel("Year")
    fig.suptitle(f"Regional mean {VARIABLE_LABEL.lower()} (all stations)", fontsize=9)
    _save(fig, outdir, f"Fig_Regional_mean{tag}")


def make_figures(res, raw, reg_table, reg_series, stations, outdir: Path, tag=""):
    for season in res["Season"].unique():
        fig_ita_grid(raw, season, stations, outdir, tag)
    fig_heatmap(res, stations, outdir, "MMK", tag)
    fig_heatmap(res, stations, outdir, "MK", tag)
    fig_agreement(res, outdir, tag)
    fig_regional(reg_series, reg_table, outdir, tag)


# =============================================================================
# 6. OUTPUT
# =============================================================================
METHODS_NOTE = """\
METHODS NOTE (edit to journal style)
------------------------------------
Trends in {var} were analysed for {n_st} stations over {y0}-{y1} for annual, JJAS,
FMAM and ONDJ series. The Mann-Kendall (MK) test (Mann, 1945; Kendall, 1975) was
applied with tie-corrected variance. Because serial correlation inflates the
false-positive rate, the Modified MK test of Hamed and Rao (1998) was applied,
in which the variance of S is multiplied by n/n* estimated from significant
autocorrelations of the ranks of the detrended series. Trend magnitude was
estimated with the Theil-Sen slope (Sen, 1968) and its {ci}% confidence interval.
The Innovative Trend Analysis (ITA; Sen, 2012) compared the sorted first and
second halves of each series (middle year dropped for odd n). ITA significance
followed Sen (2017): the ITA slope s = 2(mean_2 - mean_1)/n is significant when it
lies outside +/- z(1-alpha/2) * sigma_s, with
sigma_s = 2*sqrt(2)*sigma/(n*sqrt(pi)) * sqrt(1-rho), where sigma is the standard
deviation of the series and rho the correlation between the sorted halves
(reported as ITA_*_Sen2017 and CL90/95/99). Monte-Carlo checks on trendless
normal series (n = 31) showed that this analytic criterion rejects the null far
more often than the nominal level (~35 % at alpha = 0.05), so the decision
rule used here is an exact permutation p-value of the ITA slope ({nitaperm}
resamplings of the years), tested two-sided at the same alpha = {alpha} as MK
and MMK. The ITA slope depends only on the half means; sorting does not change it.
Multiplicity: for each season, p-values across the {n_st} stations were adjusted by
the Benjamini-Hochberg (1995) false discovery rate (q = {alpha}). Field
significance of the MK results was assessed by the Livezey and Chen (1983)
procedure with {nperm} joint resamplings of the years across stations, which
preserves inter-station correlation.
Note: the resampling destroys serial correlation, so field significance is
reported for the MK test; interpret together with the MMK results.
"""


def write_outputs(outdir: Path, res, reg_table, counts, agree, field, qc,
                  res_s=None, reg_s=None, counts_s=None, field_s=None, data=None):
    clean = lambda d: d.drop(columns=[c for c in d.columns if c.startswith("_")], errors="ignore")
    res_c = clean(res)
    res_c.to_csv(outdir / "water_balance_trend_results_all.csv", index=False, float_format="%.5g")
    xlsx = outdir / "water_balance_trend_results.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        counts.to_excel(xw, sheet_name="Summary_counts", index=False)
        agree.to_excel(xw, sheet_name="Method_agreement", index=False)
        if not field.empty:
            field.to_excel(xw, sheet_name="Field_significance", index=False)
        for season in res_c["Season"].unique():
            res_c[res_c.Season == season].drop(columns="Season").to_excel(xw, sheet_name=f"{season}_stations", index=False)
        clean(reg_table).to_excel(xw, sheet_name="Regional_mean", index=False)
        qc.to_excel(xw, sheet_name="QC_data", index=False)
        if res_s is not None:
            counts_s.to_excel(xw, sheet_name="Sens_counts", index=False)
            if field_s is not None and not field_s.empty:
                field_s.to_excel(xw, sheet_name="Sens_field_signif", index=False)
            clean(res_s).to_excel(xw, sheet_name="Sens_all_stations", index=False)
            clean(reg_s).to_excel(xw, sheet_name="Sens_regional", index=False)
        # light formatting
        for ws in xw.book.worksheets:
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 28)
    log.info("Wrote %s", xlsx.name)


# =============================================================================
# 7. MAIN
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=INPUT_DIR)
    ap.add_argument("--output", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--no-sensitivity", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    log.setLevel(logging.INFO)
    alpha = args.alpha
    if not args.input.exists():
        sys.exit(f"ERROR: input folder not found: {args.input}")
    outdir = args.output or (args.input / "trend_results")
    outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / "figures"
    figdir.mkdir(exist_ok=True)

    data = load_data(args.input)
    stations = list(next(iter(data.values())).columns)
    for s, d in data.items():
        if list(d.columns) != stations:
            log.warning("Station order/names differ in %s - using that file's own order.", s)

    qc = qc_report(data)
    flagged = qc[(qc.first_years_zero_or_100 != "")]
    for _, r in flagged.iterrows():
        log.warning("QC: %s / %s has 0 or 100 in its first years (%s)",
                    r.Season, r.Station, r.first_years_zero_or_100)

    # ---- main analysis (data as supplied) -----------------------------------
    res, raw, field = run_all(data, alpha)
    reg_table, reg_series = regional_mean(data, alpha)
    counts, agree = summary_counts(res)
    make_figures(res, raw, reg_table, reg_series, stations, figdir)

    # ---- sensitivity analysis ------------------------------------------------
    res_s = reg_s = counts_s = field_s = None
    if SENSITIVITY_MASK and not args.no_sensitivity:
        data_s = apply_mask(data, SENSITIVITY_MASK)
        res_s, raw_s, field_s = run_all(data_s, alpha, with_field=False)
        reg_s, _ = regional_mean(data_s, alpha)
        counts_s, _ = summary_counts(res_s)
        # field significance on the masked data: drop the masked stations
        keep = [c for c in stations if c not in SENSITIVITY_MASK]
        fs_rows = []
        for season, d in data_s.items():
            fs = field_significance_mk(d[keep], alpha)
            fs["Season"] = season
            fs["note"] = f"excluding masked stations: {', '.join(SENSITIVITY_MASK)}"
            fs_rows.append(fs)
        field_s = pd.DataFrame(fs_rows)
        # direction/significance changes caused by the masking
        m = res.merge(res_s, on=["Season", "Station"], suffixes=("", "_masked"))
        ch = m[(m.MMK_trend != m.MMK_trend_masked) | (m.MK_trend != m.MK_trend_masked)
               | (m.ITA_trend != m.ITA_trend_masked)]
        if len(ch):
            log.warning("Sensitivity: %d station-season results change class after masking:\n%s",
                        len(ch), ch[["Season", "Station", "MK_trend", "MK_trend_masked",
                                     "MMK_trend", "MMK_trend_masked", "ITA_trend",
                                     "ITA_trend_masked"]].to_string(index=False))
        sens_dir = figdir / "sensitivity"
        sens_dir.mkdir(exist_ok=True)
        _, rs = regional_mean(data_s, alpha)
        make_figures(res_s, raw_s, reg_s, rs, stations, sens_dir, tag="_masked")

    write_outputs(outdir, res, reg_table, counts, agree, field, qc,
                  res_s, reg_s, counts_s, field_s, data)

    first = next(iter(data.values()))
    (outdir / "methods_note.txt").write_text(METHODS_NOTE.format(
        var=VARIABLE_LABEL.lower(), n_st=len(stations), y0=first.index.min(),
        y1=first.index.max(), alpha=alpha, ci=int(100 * (1 - alpha)),
        nperm=N_PERMUTATIONS, nitaperm=N_ITA_PERM), encoding="utf-8")

    # ---- console summary -----------------------------------------------------
    pd.set_option("display.width", 200)
    print("\n=== Stations with significant trend (raw alpha = %.2f) ===" % alpha)
    print(counts[["Season", "Method", "Sig_increasing", "Sig_decreasing", "No_trend",
                  "Sig_increasing_FDR", "Sig_decreasing_FDR"]].to_string(index=False))
    if not field.empty:
        print("\n=== Field significance (MK, Livezey-Chen) ===")
        print(field.round(3).to_string(index=False))
    print(f"\nAll outputs written to: {outdir}")


if __name__ == "__main__":
    main()
