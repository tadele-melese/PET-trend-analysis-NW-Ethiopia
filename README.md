# PET trend analysis: ITA, MK/MMK test selection, FDR and field significance

Python code used for the potential evapotranspiration (PET) trend analysis of 26 stations at four
temporal scales (annual, JJAS, FMAM, ONDJ), 1984-2013 (n = 30).

## Contents

| File | Purpose |
|---|---|
| `run_all.py` | Runs everything for all temporal scales |
| `ita_analysis.py` | Innovative Trend Analysis (Sen, 2012): slope S, indicator D, confidence limits CL90/95/99, one scatter plot per station |
| `autocorrelation_decision.py` | Ljung-Box test, Durbin-Watson statistic (on regression residuals), MK, MMK (Hamed and Rao, 1998) and the MK/MMK decision rule |
| `fdr_field_significance.py` | Benjamini-Hochberg / Benjamini-Yekutieli FDR control, permutation p-values for ITA, field significance |

## Usage

```bash
pip install -r requirements.txt
# put the input files in data/  (Year + one column per station)
python run_all.py
```

Expected input files (names can be changed in `run_all.py`): `Annual_PET_mean.csv`,
`Seasonal_PET_JJAS.csv`, `Seasonal_PET_FMAM.csv`, `Seasonal_PET_ONDJ.csv`.
The scripts can also be run one file at a time after editing `INPUT_CSV` at the top of each script.

## Outputs (`results/`)

* `ITA/<file>/ITA_Results.csv` and `ITA_<station>.png` - ITA statistics and plots
* `Autocorr_<file>.csv` - DW, Ljung-Box Q and p, lag-1 r, MK p, MMK p, selected test and its p-value
* `Table2_autocorrelation.csv` - Table 2 (DW, Ljung-Box Q and p for all scales)
* `FDR_FieldSignificance_<file>.csv` - MK / ITA p-values and BH / BY decisions; field-significance summary is printed

## Methods in brief

* **ITA**: the series is split into two halves (n = 15 each), each half is sorted;
  S = 2(M2 - M1)/N, D = (1/n) * sum(10 (y_i - x_i) / M1);
  CL = +/- z * sigma_S, sigma_S = 2*sqrt(2) * sigma * sqrt(1 - rho) / (N * sqrt(N)), z = 1.645, 1.96, 2.576.
  Sen's limits are liberal (they flag about 76% of trend-free series at the nominal 5% level when N = 30),
  so significance statements should rely on MK/MMK or on the permutation p-values.
* **Decision rule**: Ljung-Box (10 lags) p < 0.05 -> MMK; otherwise MK. Both are always reported.
  Durbin-Watson is computed on the residuals of a regression on time (never on the differenced series)
  with Monte Carlo p-values and is a complementary lag-1 diagnostic.
* **MMK**: Hamed and Rao (1998) variance correction using ranks of the Sen-slope-detrended series and all lags
  with |rho_k| > 1.96/sqrt(n); if the correction factor is non-positive it falls back to MK.
* **FDR / field significance**: BH (q = 0.05) per scale and pooled; BY as a conservative check; field significance by
  permuting years jointly for all stations (5000 permutations, fixed seed). The permutation does not preserve
  serial correlation.

## References

Benjamini and Hochberg (1995) J. R. Stat. Soc. B 57:289-300; Benjamini and Yekutieli (2001) Ann. Stat. 29:1165-1188;
Hamed and Rao (1998) J. Hydrol. 204:182-196; Livezey and Chen (1983) Mon. Weather Rev. 111:46-59;
Sen (2012) Hydrol. Sci. J. / Water Resour. Manage. - please cite the exact ITA reference used in the paper;
Wilks (2016) Bull. Am. Meteorol. Soc. 97:2263-2273.
