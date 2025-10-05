import os
import time
import tempfile
import numpy as np
import pandas as pd
import lightkurve as lk
from tqdm import tqdm
from astropy.utils.data import clear_download_cache  

#setting
CSV_PATH = r"C:\Users\DL\Desktop\NASA2025\cumulative_2025.09.24_03.35.35.csv"
OUT_DIR  = r"D:\NASA2025"       
MISSION  = "Kepler"              # "Kepler" | "K2" | "TESS"
CADENCE  = "long"                # "long" or "short"
MAX_FILES_PER_TARGET = None      # (None=full)
WINDOW_LENGTH = 401              # flatten 
SLEEP_SEC = 0.4                  # API 
#

#read csv
os.makedirs(OUT_DIR, exist_ok=True)
catalog = pd.read_csv(CSV_PATH, comment="#")
if "kepid" not in catalog.columns:
    raise ValueError("wrong colume！")

kepids = (
    catalog["kepid"].dropna().astype(int).astype(str).unique()
)
print(f"Total: {len(kepids)}  KIC ID。")

def save_star_as_csv(kic_id: str,
                     mission=MISSION,
                     cadence=CADENCE,
                     max_files=MAX_FILES_PER_TARGET,
                     window_length=WINDOW_LENGTH) -> bool:
    target = f"KIC {kic_id}"

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # LightCurve download
            sr = lk.search_lightcurve(target, mission=mission, cadence=cadence)
            if len(sr) == 0:
                print(f"{target}: can not find the file")
                return False

            if max_files is not None:
                sr = sr[:max_files]

            # 
            lcc = sr.download_all(download_dir=tmpdir, quality_bitmask='hard')
            if lcc is None or len(lcc) == 0:
                print(f"{target}: not valid data")
                return False

            # stitch and prepocess
            lc = (lcc.stitch()
                     .remove_nans()
                     .remove_outliers()
                     .normalize()
                     .flatten(window_length=window_length))

            # 
            df_out = pd.DataFrame({
                "time_bkjd": lc.time.value.astype("float32"),
                "flux_norm": lc.flux.value.astype("float32")
            })

            out_path = os.path.join(OUT_DIR, f"{kic_id}.csv")
            df_out.to_csv(out_path, index=False)
            return True

        except Exception as e:
            print(f"{target}: wrong -> {e}")
            return False
        finally:

            try:
                clear_download_cache()
            except Exception:
                pass

for i, kid in enumerate(tqdm(kepids, desc="Export CSV"), 1):
    ok = save_star_as_csv(kid)
    status = "OK" if ok else "Failed"
    print(f"[{i}/{len(kepids)}] KIC {kid}: {status}")
    time.sleep(SLEEP_SEC)

print("complete")