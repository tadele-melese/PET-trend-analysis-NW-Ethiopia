"""
Innovative Trend Analysis (ITA, Sen 2012) for many stations at once.

Input : a CSV with a 'Year' column and one column per station
        (e.g. Seasonal_PET_FMAM.csv)
Output: - ITA_Results.csv  (S, D, confidence limits, Trend)
        - one ITA scatter plot (PNG) per station

Usage:  python ita_analysis.py
        (edit the SETTINGS block below first)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# SETTINGS  -> change these
# ----------------------------------------------------------------------
INPUT_CSV = "Seasonal_PET_FMAM.csv"
OUT_DIR = "ITA_output"
RESULTS_CSV = "ITA_Results.csv"
TREND_LEVEL = 90          # confidence level (90, 95 or 99) used for the Trend label
N_USED = 30               # years used: first 30 (1984-2013); 2 halves of 15
DPI = 300
# ----------------------------------------------------------------------

# z-values (rounded as in the original results)
Z = {90: 1.645, 95: 1.96, 99: 2.576}


def ita(series, n_used=None):
    """Return dict with sorted halves and ITA statistics for one station."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    if n_used is not None:
        x = x[:n_used]
    N = len(x)
    if N % 2 == 1:                       # need an even number of values
        x = x[:-1]
        N -= 1
    n = N // 2                           # length of each half

    first = np.sort(x[:n])               # first half, sorted ascending
    second = np.sort(x[n:])              # second half, sorted ascending
    diff = second - first

    # Slope (Sen 2012):  S = 2 * (mean2 - mean1) / N
    S = 2.0 * (second.mean() - first.mean()) / N

    # Relative trend indicator (%):  D = (1/n) * sum( 10 * (y_i - x_i) / mean(x) )
    D = np.mean(10.0 * diff / first.mean())

    # Confidence limits of the slope (Sen 2012)
    rho = np.corrcoef(first, second)[0, 1]
    sigma = x.std(ddof=1)
    sigma_s = 2.0 * np.sqrt(2.0) * sigma * np.sqrt(1.0 - rho) / (N * np.sqrt(N))

    out = {"first": first, "second": second, "diff": diff, "S": S, "D": D}
    for lvl, z in Z.items():
        out[f"CL{lvl}_lower"] = -z * sigma_s
        out[f"CL{lvl}_upper"] = z * sigma_s
    return out


def classify(S, lower, upper):
    if S > upper:
        return "Increasing"
    if S < lower:
        return "Decreasing"
    return "No Trend"


def ita_plot(name, res, path, dpi=300):
    first, second, diff = res["first"], res["second"], res["diff"]

    lo = min(first.min(), second.min())
    hi = max(first.max(), second.max())
    pad = 0.0 * (hi - lo)
    lim = (lo - pad, hi + pad)
    xx = np.linspace(lim[0] - 1000, lim[1] + 1000, 10)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.plot(xx, xx, "k-", lw=2.5, label="No Trend (1:1)")
    ax.plot(xx, xx * 1.05, color="green", ls="--", lw=2, label="±5%")
    ax.plot(xx, xx * 0.95, color="green", ls="--", lw=2)
    ax.plot(xx, xx * 1.10, color="purple", ls=":", lw=2.5, label="±10%")
    ax.plot(xx, xx * 0.90, color="purple", ls=":", lw=2.5)

    sc = ax.scatter(first, second, c=diff, cmap="coolwarm", s=70,
                    edgecolor="black", linewidth=1, alpha=0.9, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Difference (Second - First Half)", fontsize=11)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("First Half (Sorted)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Second Half (Sorted)", fontsize=14, fontweight="bold")
    ax.set_title(f"ITA - {name}", fontsize=16, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="upper left", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    stations = [c for c in df.columns if c.lower() != "year"]

    rows = []
    for st in stations:
        res = ita(df[st], n_used=N_USED)
        trend = classify(res["S"],
                         res[f"CL{TREND_LEVEL}_lower"],
                         res[f"CL{TREND_LEVEL}_upper"])
        rows.append({
            "Station": st,
            "S": res["S"], "D": res["D"],
            "CL90_lower": res["CL90_lower"], "CL90_upper": res["CL90_upper"],
            "CL95_lower": res["CL95_lower"], "CL95_upper": res["CL95_upper"],
            "CL99_lower": res["CL99_lower"], "CL99_upper": res["CL99_upper"],
            "Trend": trend,
        })
        ita_plot(st, res, os.path.join(OUT_DIR, f"ITA_{st}.png"), dpi=DPI)

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(OUT_DIR, RESULTS_CSV), index=False)
    print(results.round(3).to_string(index=False))
    print(f"\nSaved results and {len(stations)} plots in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
