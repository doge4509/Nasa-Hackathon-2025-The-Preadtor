import numpy as np
import matplotlib.pyplot as plt
path = r"C:\Users\DL\Desktop\K\折相\K00771.01.npz"
z = np.load(path, allow_pickle=True)

print("檔案內容 keys:", list(z.keys()))

# 檢查 shape
print("X.shape:", z["X"].shape)        # (2, L)
print("xi.shape:", z["xi"].shape)      # (L,)
print("mask.shape:", z["mask"].shape)  # (L,)


X     = z["X"].astype(np.float32)          # [2, L]
xi    = z["xi"].astype(np.float32)         # [L]
mask  = z["mask"].astype(np.float32)       # [L]
koi   = str(np.asarray(z["kepoi_name"]).item()) if "kepoi_name" in z else "UNKNOWN"
kepid = int(np.asarray(z["kepid"]).item()) if "kepid" in z else -1

# 檢查 meta
print("kepoi_name:", z["kepoi_name"])
print("kepid:", z["kepid"])
print("period:", z["period"])
print("t0:", z["t0"])
print("duration_hr:", z["duration_hr"])
print("disposition:", z["disposition"])
print("y (label):", z["y"])



# 看輔助參數
for k in ["koi_depth","koi_prad","koi_teq","koi_insol","koi_steff","koi_slogg","koi_srad","ra","dec","kepmag"]:
    print(f"{k}: {z[k]}")

valid = (mask > 0.5)
valid &= np.isfinite(X[0]) & np.isfinite(X[1]) & np.isfinite(xi)

plt.figure(figsize=(8,4))
plt.scatter(xi[valid], X[0][valid], s=6, alpha=0.35, label="ch0 (raw)")
plt.plot(xi[valid],   X[0][valid], linewidth=0.8, alpha=0.6)

plt.scatter(xi[valid], X[1][valid], s=6, alpha=0.35, label="ch1 (smoothed)")
plt.plot(xi[valid],   X[1][valid], linewidth=0.8, alpha=0.6)

plt.axvline(0.0, linestyle="--", linewidth=0.8, alpha=0.5)  # 食甚中心
plt.xlabel("Phase (cycles)")
plt.ylabel("Flux (robust scaled)")
plt.title(f"{koi} (kepid={kepid})")
plt.legend()
plt.tight_layout()
plt.show()