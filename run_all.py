"""
Run the whole analysis for every temporal scale.

1. ITA statistics and one scatter plot per station        -> results/ITA/<file>/
2. Serial-autocorrelation diagnostics + MK/MMK decision    -> results/Autocorr_<file>.csv
3. FDR control and field significance (MK and ITA)         -> results/FDR_FieldSignificance_<file>.csv
4. Table 2 (wide format, all scales)                       -> results/Table2_autocorrelation.csv

Put the input CSV files (Year + one column per station) in the folder  data/  and run:
    python run_all.py
"""
import os
import pandas as pd
import ita_analysis as ita
import autocorrelation_decision as ac
import fdr_field_significance as fdr

# ---------------- SETTINGS ----------------
DATA_DIR = "data"
RESULTS = "results"
FILES = {                       # label used in Table 2 : file name
    "Annual": "Annual_PET_mean.csv",
    "JJAS": "Seasonal_PET_JJAS.csv",
    "FMAM": "Seasonal_PET_FMAM.csv",
    "ONDJ": "Seasonal_PET_ONDJ.csv",
}
# ------------------------------------------

os.makedirs(RESULTS, exist_ok=True)
for module in (ac, fdr):
    module.OUT_DIR = RESULTS

table2 = None
for label, f in FILES.items():
    path = os.path.join(DATA_DIR, f)
    if not os.path.exists(path):
        print(f"!! {path} not found - skipped")
        continue
    name = os.path.splitext(f)[0]
    print("=" * 70, f"\n{label}  ({f})\n" + "=" * 70)

    ita.INPUT_CSV = path
    ita.OUT_DIR = os.path.join(RESULTS, "ITA", name)
    ita.main()

    ac.INPUT_CSV = path
    ac.main()

    fdr.INPUT_CSV = path
    fdr.main()

    a = pd.read_csv(os.path.join(RESULTS, f"Autocorr_{name}.csv"))
    part = a[["Station", "DW", "LB_Q", "LB_p"]].rename(
        columns={"DW": f"{label}_DW", "LB_Q": f"{label}_LB_Q", "LB_p": f"{label}_LB_p"})
    table2 = part if table2 is None else table2.merge(part, on="Station", how="outer")

if table2 is not None:
    table2.to_csv(os.path.join(RESULTS, "Table2_autocorrelation.csv"), index=False)
    print("\nSaved results/Table2_autocorrelation.csv")
