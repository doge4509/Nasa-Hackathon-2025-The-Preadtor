import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter

path = r"C:\Users\DL\Downloads\cumulative_2025.09.24_03.35.16.csv"
in_dir  = r"C:\Users\DL\Desktop\K\未折相"
out_dir = r"C:\Users\DL\Desktop\K\折相"

def read_koi_table(p):
    df = pd.read_csv(p, comment="#", sep="\t", engine="python")
    if df.shape[1] == 1:
        df = pd.read_csv(p, comment="#", sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df

def fold_to_fixed_length(time_bkjd, flux, P, t0, duration_hr=None,
                         L=512, k_window=3.0, smooth_bins=64):

    # 1) phase: [-0.5,0.5)
    phase = ((time_bkjd - t0) / P + 0.5) % 1.0 - 0.5

    # 2) window
    if duration_hr is not None and np.isfinite(duration_hr) and duration_hr > 0:
        W = float(k_window * (duration_hr/24.0) / P)  
        W = float(np.clip(W, 0.03, 0.2))              
    else:
        W = 0.1

    m = (phase >= -W) & (phase <= +W)
    if m.sum() < 10:
        m = slice(None)

    ph_w = phase[m]
    fx_w = flux[m]

    # 3) resampling
    xi = np.linspace(-W, +W, L)
    order = np.argsort(ph_w)
    f = interp1d(ph_w[order], fx_w[order], kind="linear",
                 bounds_error=False, fill_value=np.nan, assume_sorted=True)
    ch0 = f(xi) 

    # 4) median_filter
    k = max(3, L // smooth_bins)
    ch1 = median_filter(ch0, size=k)

    # 5) robust 
    def robust_scale(x):
        med = np.nanmedian(x)
        iqr = np.nanpercentile(x, 75) - np.nanpercentile(x, 25)
        if not np.isfinite(iqr) or iqr == 0:
            iqr = np.nanstd(x) if np.isfinite(np.nanstd(x)) else 1.0
        y = (x - med) / iqr
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    ch0 = robust_scale(ch0)
    ch1 = robust_scale(ch1)

    # 6) mask
    mask = (~np.isnan(f(xi))).astype(np.float32)

    X = np.stack([ch0, ch1], axis=0).astype(np.float32)  # [2, L]
    return X, xi, mask


df = read_koi_table(path)

# 
needed_cols = ["kepid",
    "kepoi_name", "koi_period", "koi_time0bk", "koi_duration",
    "koi_disposition", "koi_depth", "koi_prad", "koi_teq", "koi_insol",
    "koi_steff", "koi_slogg", "koi_srad", "ra", "dec", "koi_kepmag"
]

df_needed = df[needed_cols]
#print(df_needed.head())

df_cat = df


#target_list = [
#    "K00752.01"]
target_list = df_cat["kepoi_name"].dropna().unique().tolist()
for koi in target_list:
    row = df_cat.loc[df_cat["kepoi_name"] == koi]
    if row.empty:
        print(f"[WARN] catalog can not find {koi}")
        continue

    row = row.iloc[0]
    P  = float(row["koi_period"])
    t0 = float(row["koi_time0bk"])
    dur = row["koi_duration"]
    kepid = int(row["kepid"])  # 

    print(f"[INFO] {koi} (kepid={kepid}), P={P}, t0={t0}")

    # === name with kepid  ===
    in_csv = os.path.join(in_dir, f"{kepid}.csv")
    if not os.path.exists(in_csv):
        print(f"[WARN]: {in_csv}")
        continue

    df_lc = pd.read_csv(in_csv)
    time = df_lc["time_bkjd"].values
    flux = df_lc["flux_norm"].values

    # === fold_to_fixed_length ===
    X, xi, mask = fold_to_fixed_length(time, flux, P, t0, duration_hr=dur)

    # === save file ===
    out_npz = os.path.join(out_dir, f"{koi}.npz")
    #np.savez(out_npz, X=X, xi=xi, mask=mask,
    #         kepoi_name=koi, kepid=kepid, period=P, t0=t0, duration=dur)
    
    np.savez(out_npz,
        X=X.astype(np.float32),
        xi=xi.astype(np.float32),
        mask=mask.astype(np.float32),

        # basic meta
        kepoi_name=str(koi),
        kepid=np.int64(kepid),
        period=np.float32(P),
        t0=np.float32(t0),
        duration_hr=np.float32(dur if pd.notnull(dur) else np.nan),

        # catalog 
        
        disposition=str(row["koi_disposition"]),
        koi_depth=np.float32(row["koi_depth"]) if pd.notnull(row["koi_depth"]) else np.nan,
        koi_prad=np.float32(row["koi_prad"]) if pd.notnull(row["koi_prad"]) else np.nan,
        koi_teq=np.float32(row["koi_teq"]) if pd.notnull(row["koi_teq"]) else np.nan,
        koi_insol=np.float32(row["koi_insol"]) if pd.notnull(row["koi_insol"]) else np.nan,
        koi_steff=np.float32(row["koi_steff"]) if pd.notnull(row["koi_steff"]) else np.nan,
        koi_slogg=np.float32(row["koi_slogg"]) if pd.notnull(row["koi_slogg"]) else np.nan,
        koi_srad=np.float32(row["koi_srad"]) if pd.notnull(row["koi_srad"]) else np.nan,
        ra=np.float32(row["ra"]) if pd.notnull(row["ra"]) else np.nan,
        dec=np.float32(row["dec"]) if pd.notnull(row["dec"]) else np.nan,
        kepmag=np.float32(row["koi_kepmag"]) if pd.notnull(row["koi_kepmag"]) else np.nan,

   
    y=np.int64(
        1 if row["koi_disposition"] == "CONFIRMED" else
        0 if row["koi_disposition"] == "CANDIDATE" else
        -1 if row["koi_disposition"] == "FALSE POSITIVE" else
        -2   #
    ))
    

    print(f"[SAVE] {out_npz} complete X.shape={X.shape}")

    # === Quickview ===
    plt.figure(figsize=(8,4))
    plt.scatter(xi, X[0], s=4, alpha=0.4)
    plt.plot(xi, X[0], linestyle='-', linewidth=0.8, alpha=0.6, label="ch0")
    plt.scatter(xi, X[1], s=4, alpha=0.4)
    plt.plot(xi, X[1], linestyle='-', linewidth=0.8, alpha=0.6, label="ch1")
    plt.xlabel("Phase (cycles)")
    plt.ylabel("Flux (robust scaled)")
    plt.title(f"{koi} (kepid={kepid})")
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{koi}.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[SAVE] Quickview: {out_png}")


